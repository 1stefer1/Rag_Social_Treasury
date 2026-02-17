import atexit
import copy
import inspect
import logging
import logging.config
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

# Базовый конфиг логирования, сохраняем оригинальный формат для ELK
# и настраиваем московское время
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",  # Формат даты без часового пояса
        }
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "async_file_handler": {
            # Будет заменен в коде настройки логгера
            "class": "logging.FileHandler",
            "level": "INFO",
            "filename": "logs/logger.log",
            "formatter": "standard",
            "encoding": "utf-8",
        },
        "async_warning_handler": {
            # Будет заменен в коде настройки логгера
            "class": "logging.FileHandler",
            "level": "WARNING",
            "filename": "logs/warnings.log",
            "formatter": "standard",
            "encoding": "utf-8",
        },
        "async_debug_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": "logs/debug.log",
            "formatter": "standard",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "async_file_handler", "async_warning_handler"],
            "level": "INFO",
            "propagate": False,
        }
    },
}


class AsyncLogWriter:
    """
    Класс для асинхронной записи логов в файлы.
    Использует отдельный поток для записи из очереди сообщений.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Реализация паттерна Singleton для лог-писателя."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = AsyncLogWriter()
                cls._instance.start()
            return cls._instance

    def __init__(self):
        """Инициализация лог-писателя."""
        # Устанавливаем московское время
        os.environ["TZ"] = "Europe/Moscow"
        try:
            time.tzset()
        except AttributeError:
            pass  # Windows не поддерживает tzset

        # Создаем директорию для логов
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Очередь сообщений для записи
        self.queue = queue.Queue(maxsize=10000)  # Максимум 10000 сообщений в очереди

        # Кэш открытых файлов логов
        self.files = {}
        self.formatters = {}

        # Флаг работы потока и сам поток
        self.running = False
        self.worker_thread = None

    def start(self):
        """Запускает поток обработки логов."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._process_logs, daemon=True
            )
            self.worker_thread.start()

    def stop(self):
        """Останавливает поток обработки логов."""
        if self.running:
            self.running = False
            if self.worker_thread and self.worker_thread.is_alive():
                # Даем потоку время закончить обработку очереди
                self.worker_thread.join(timeout=3.0)

            # Закрываем все открытые файлы
            for file_obj in self.files.values():
                try:
                    file_obj.close()
                except:
                    pass
            self.files.clear()

    def enqueue(self, record, filename, level=logging.INFO):
        """
        Помещает запись лога в очередь на запись.

        Args:
            record: Запись лога
            filename: Имя файла для записи
            level: Уровень логгирования (для выбора форматтера)
        """
        try:
            # Создаем кортеж с информацией для записи
            log_item = (record, filename, level)
            # Используем неблокирующую постановку в очередь с таймаутом
            # для случаев, когда очередь заполнена
            self.queue.put(log_item, block=True, timeout=0.1)
        except queue.Full:
            # Если очередь переполнена, пишем в stderr
            sys.stderr.write(
                f"Очередь логов переполнена, сообщение потеряно: {record}\n"
            )
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Ошибка при добавлении в очередь логов: {e}\n")
            sys.stderr.flush()

    def _get_file(self, filename):
        """Получает файловый объект для записи, создавая его при необходимости."""
        if filename not in self.files:
            # Создаем директорию при необходимости
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            # Открываем файл в режиме добавления с буферизацией
            self.files[filename] = open(
                filename, mode="a", encoding="utf-8", buffering=8192
            )
        return self.files[filename]

    def _get_formatter(self, level):
        """Получает форматтер для нужного уровня логгирования."""
        if level not in self.formatters:
            format_str = LOGGING_CONFIG["formatters"]["standard"]["format"]
            date_fmt = LOGGING_CONFIG["formatters"]["standard"].get(
                "datefmt", "%Y-%m-%d %H:%M:%S"
            )
            self.formatters[level] = logging.Formatter(format_str, date_fmt)
        return self.formatters[level]

    def _process_logs(self):
        """Основной метод обработки логов из очереди."""
        last_flush_time = time.time()

        while self.running or not self.queue.empty():
            try:
                # Получаем запись из очереди с таймаутом
                try:
                    record, filename, level = self.queue.get(timeout=0.5)
                except queue.Empty:
                    # Если очередь пуста, делаем flush всех файлов если прошло достаточно времени
                    current_time = time.time()
                    if (
                        current_time - last_flush_time > 2.0
                    ):  # Каждые 2 секунды делаем flush
                        for file_obj in self.files.values():
                            file_obj.flush()
                        last_flush_time = current_time
                    continue

                # Получаем файл и форматтер
                file_obj = self._get_file(filename)
                formatter = self._get_formatter(level)

                # Форматируем и записываем сообщение
                formatted_message = formatter.format(record) + "\n"
                file_obj.write(formatted_message)

                # Отмечаем задачу как выполненную
                self.queue.task_done()

                # Если очередь пуста, сразу делаем flush
                if self.queue.empty():
                    file_obj.flush()
                    last_flush_time = time.time()

            except Exception as e:
                # Логируем ошибки в stderr
                sys.stderr.write(f"Ошибка при записи лога: {e}\n")
                sys.stderr.flush()
                time.sleep(0.1)  # Пауза чтобы не нагружать CPU при ошибках


# --- Вспомогательная функция для генерации имени файла с датой ---
def get_daily_log_filename(base_name: str) -> str:
    today = time.strftime("%Y-%m-%d")
    name, ext = os.path.splitext(base_name)
    return f"{name}_{today}{ext}"


class AsyncHandler(logging.Handler):
    """
    Обработчик, помещающий логи в очередь для асинхронной записи.
    """

    def __init__(self, filename, level=logging.INFO, daily_rotate=True):
        super().__init__(level=level)
        self.base_filename = filename
        self.log_level = level
        self.writer = AsyncLogWriter.get_instance()
        self.daily_rotate = daily_rotate
        self._current_filename = None
        self._current_date = None

    def emit(self, record):
        """Помещает запись в очередь для асинхронной обработки."""
        try:
            filename = self.base_filename
            if self.daily_rotate:
                today = time.strftime("%Y-%m-%d")
                if self._current_date != today:
                    self._current_date = today
                    self._current_filename = get_daily_log_filename(self.base_filename)
                filename = self._current_filename
            else:
                filename = self.base_filename
            self.writer.enqueue(record, filename, self.log_level)
        except Exception:
            self.handleError(record)


def get_async_handlers(logger_name):
    """
    Создает и возвращает асинхронные обработчики логов.

    Args:
        logger_name: Имя логгера (используется для имени файла)

    Returns:
        tuple: (file_handler, warning_handler) - асинхронные обработчики
    """
    # Директория для логов
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Пути к файлам
    log_file = str(log_dir / f"{logger_name}.log")
    warning_file = str(log_dir / "warnings.log")

    # Создаем асинхронные обработчики с ежедневной ротацией
    file_handler = AsyncHandler(
        filename=log_file, level=logging.INFO, daily_rotate=True
    )
    warning_handler = AsyncHandler(
        filename=warning_file, level=logging.WARNING, daily_rotate=True
    )

    debug_file = str(log_dir / "debug.log")
    debug_handler = AsyncHandler(
        filename=debug_file, level=logging.DEBUG, daily_rotate=True
    )

    return file_handler, warning_handler, debug_handler


def setup_shutdown_handlers():
    """Настраивает корректную остановку логирования при завершении приложения."""

    def shutdown_handler():
        """Останавливает асинхронное логирование."""
        writer = AsyncLogWriter.get_instance()
        writer.stop()

    # Регистрируем обработчик для штатного завершения
    atexit.register(shutdown_handler)

    # Регистрируем обработчики сигналов для корректного завершения
    def signal_handler(signum, frame):
        shutdown_handler()
        # Вызываем оригинальный обработчик, если он был
        original_handler = signal.getsignal(signum)
        if callable(original_handler) and original_handler is not signal_handler:
            original_handler(signum, frame)

    # Регистрируем обработчики сигналов
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, signal_handler)
        except (OSError, ValueError):
            pass  # Игнорируем, если сигнал не поддерживается


# Настраиваем корректное завершение при выходе
setup_shutdown_handlers()


def setup_logger() -> logging.Logger:
    """
    Настраивает и возвращает асинхронный логгер для текущего модуля.
    Логи будут записываться в отдельном потоке, не блокируя основное приложение.

    Returns:
        logging.Logger: Настроенный логгер для модуля вызывающей стороны
    """
    # Получаем имя файла вызывающей стороны
    caller_filename = inspect.stack()[1].filename
    logger_name = os.path.splitext(os.path.basename(caller_filename))[0]

    # Проверяем, не настроен ли уже логгер
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    # Создаем копию конфигурации, чтобы не изменять оригинал
    log_config = copy.deepcopy(LOGGING_CONFIG)

    # Применяем базовую конфигурацию
    logging.config.dictConfig(log_config)

    # Получаем асинхронные обработчики
    file_handler, warning_handler, debug_handler = get_async_handlers(logger_name)

    # Заменяем стандартные обработчики на асинхронные
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[
        :
    ]:  # Копируем список, чтобы безопасно модифицировать
        if isinstance(handler, logging.FileHandler) and not hasattr(
            handler, "async_handler_flag"
        ):
            logger.removeHandler(handler)
            if handler.level == logging.WARNING:
                logger.addHandler(warning_handler)
            elif handler.level == logging.DEBUG:
                logger.addHandler(debug_handler)
            else:
                logger.addHandler(file_handler)
    logger.addHandler(debug_handler)

    return logger


logger = setup_logger()

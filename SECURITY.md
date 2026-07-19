# Security policy

## Reporting

Не создавайте публичный issue для уязвимости или найденного секрета. Используйте приватный security advisory владельца GitHub-репозитория либо другой приватный канал, указанный владельцем проекта.

Сообщите затронутую версию, сценарий воспроизведения, impact и минимальные безопасные детали. Не прикладывайте реальные customer documents, credentials или conversation logs.

## Scope and operational rules

- `.env`, API keys, Telegram tokens, индексы и пользовательские документы не должны попадать в Git или Docker context.
- Retrieved documents, user input и CRM notes считаются недоверенными и могут содержать prompt injection.
- Production требует API auth; TLS, rate limiting и network policy должны обеспечиваться ingress/platform layer.
- При утечке секрета сначала отзовите и ротируйте его, затем очистите историю Git. Обычного удаления из последнего commit недостаточно.

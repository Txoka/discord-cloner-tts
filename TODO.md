# TODO
- Handle `ClientException: Already connected to a voice channel` by reusing or moving the existing voice connection.
- Defer `/join` responses (or use followup) to avoid `Unknown interaction` errors when responses are late.

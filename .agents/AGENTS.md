# Manufacturing Copilot — Agent Contract

1. Never claim a step succeeded without showing the command output that proves it
   (test pass, eval score, `git log`/`git ls-remote` for pushes, a screenshot for UI).
   If you can't verify it in this session, say so explicitly instead of asserting success.
2. The database layer is read-only-by-stored-procedure. Never write raw SQL against
   DB tables. Never add INSERT/UPDATE/DELETE anywhere. If a task seems to require
   writing to the database, stop and ask — don't route around the constraint.
3. Every new dependency gets a pinned version. Every new tool/skill ships with at least
   one passing test or eval before it's considered done.
4. No secrets, connection strings, or API keys in committed code. Read them from
   environment/config, and add the relevant entries to `.gitignore` before your first
   commit, not after.
5. Before editing a file you haven't seen this session, read it. Don't infer its
   contents from the architecture summary in chat history — the summary can be stale
   or wrong.
6. If a previous session's "Summary of Work" claims something is done (a push, a file,
   a deployment), verify it yourself before building on top of it.

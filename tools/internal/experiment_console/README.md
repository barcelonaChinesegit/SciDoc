# Internal experiment console

This optional maintainer tool monitors experiments sharing GPUs and hosts the
collaborative QA review interface. Benchmark users do not need it to validate
or score stored model results. Its source moved from the former top-level
`task_queue_web/` to `tools/internal/experiment_console/web/`.

```bash
cd tools/internal/experiment_console/web
npm ci
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy npm test
```

The queue link and home-page card are shown only to administrators. Existing
authenticated read endpoints and administrator write permissions are preserved.
Deploy with the updated Web systemd unit and verify the public HTTPS entry:

```bash
systemctl --user restart pku-task-queue-web.service
PYTHONPATH=src python -m pku_qa.workflows.operations.check_web_stack
```

See [deployment documentation](../../../docs/current/WEB_CONSOLE.md).

For the supported scoring entry point, use the
[evaluation Quick Start](../../../evaluation/README.md#quick-start).
The console's internal inference/binary-Judge reports are separate from the
public sxz v4 scorer; launching the Web service is not required for replay.

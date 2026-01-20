generate python code from root of repository. they end up in generated

    python -m grpc.tools.protoc -I protos --python_betterproto2_out=src/nf_robot/generated src/nf_robot/protos/*.proto

https://betterproto.github.io/python-betterproto2/getting-started/

    from generated.nf.telemetry import TelemetryBatchUpdate
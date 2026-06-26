# zed/generate_fusion_template.py
# Run once with system python to create zed/zed360_template.json
# ZED SDK 5.3.1 verified

import pyzed.sl as sl
import sys

def make_template(out_path: str):

    configs = []

    # Stream ports use the 31000 band (the real ZED-X ROS2 driver binds 30000-30003).
    # cam_c (1003) is the optional centered overhead camera used by the 3-cam sweep.
    for serial, stream_port, publish_port in [
        (1001, 31000, 30010),
        (1002, 31002, 30012),
        (1003, 31004, 30014),
    ]:
        fc = sl.FusionConfiguration()
        fc.serial_number = serial

        # Input: this camera streams from Isaac on localhost:stream_port
        input_type = sl.InputType()
        input_type.set_from_stream("127.0.0.1", stream_port)
        fc.input_type = input_type

        # Communication: how Fusion's internal pub/sub talks to this sender
        # publish_port must be even and different from stream_port
        comm = sl.CommunicationParameters()
        comm.set_for_local_network(publish_port)
        fc.communication_parameters = comm

        # Identity pose — make_fusion_config.py overwrites this with GT values
        pose = sl.Transform()
        pose.set_identity()
        fc.pose = pose

        configs.append(fc)

    sl.write_configuration_file(
        out_path,
        configs,
        sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP,
        sl.UNIT.METER
    )
    print(f"Template written to: {out_path}")
    print("Verify it has three entries (1001/1002/1003), then commit as "
          "zed/zed360_template.json")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "zed/zed360_template.json"
    make_template(out)

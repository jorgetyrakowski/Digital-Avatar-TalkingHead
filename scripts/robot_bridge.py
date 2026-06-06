#!/usr/bin/env python3
"""
robot_bridge.py — Run this INSIDE the robotic_agent_system container.

Subscribes to /object_query_choice and /task_reply, then POSTs events
to the avatar server so it can speak them to the user.

Usage (inside container):
  source /opt/ros/humble/setup.bash
  python3 robot_bridge.py

Environment variables:
  AVATAR_SERVER_URL  URL of the avatar server as seen from inside the container
                     Default: https://172.17.0.1:8010  (Docker bridge gateway on Linux)
                     Use https://host.docker.internal:8010 on Docker Desktop (Mac/Windows)

Note: the avatar server runs over HTTPS with a self-signed certificate, so the
bridge skips certificate verification when posting events.
"""
import json
import os
import ssl
import urllib.request
import urllib.error
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

AVATAR_SERVER_URL = os.environ.get("AVATAR_SERVER_URL", "https://172.17.0.1:8010")
ROBOT_EVENT_URL   = f"{AVATAR_SERVER_URL}/robot_event"

# Avatar uses a self-signed cert — accept it (internal Docker-host traffic only).
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE


class RobotBridge(Node):
    def __init__(self):
        super().__init__("robot_bridge")
        self.create_subscription(String, "/object_query_choice", self._on_choice, 10)
        self.create_subscription(String, "/task_reply",          self._on_task_reply, 10)
        self.get_logger().info(f"robot_bridge started — posting to {ROBOT_EVENT_URL}")

    def _post(self, topic: str, data: dict):
        payload = json.dumps({"topic": topic, "data": data}).encode()
        req = urllib.request.Request(
            ROBOT_EVENT_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5, context=_SSL_CTX)
            self.get_logger().info(f"Posted {topic} to avatar server")
        except urllib.error.URLError as e:
            self.get_logger().error(f"Failed to reach avatar server: {e.reason}")
        except Exception as e:
            self.get_logger().error(f"POST error: {e}")

    def _on_choice(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in /object_query_choice: {e}")
            return
        self._post("/object_query_choice", data)

    def _on_task_reply(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in /task_reply: {e}")
            return
        self._post("/task_reply", data)


def main():
    rclpy.init()
    node = RobotBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ExternalShutdownException: SIGTERM (e.g. start.sh restarting the
        # bridge) already shut down the rcl context — exit quietly.
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # context already shut down


if __name__ == "__main__":
    main()

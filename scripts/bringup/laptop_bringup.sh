#!/bin/bash

function confirm_devices {
    devices=$(adb devices)
    
    echo "List of devices:"
    echo "$devices"
    
    read -p "Is your oculus device connected? (y/n): " confirmation
    
    if [ "$confirmation" != "y" ] && [ "$confirmation" != "Y" ]; then
	return 1
    fi
}

# ensure GUI window is accessible from container
echo -e "Set Docker Xauth for x11 forwarding \n"

export DOCKER_XAUTH=/tmp/.docker.xauth
rm $DOCKER_XAUTH
touch $DOCKER_XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $DOCKER_XAUTH nmerge -

echo -e "Set environment variables from parameters file \n"

ROOT_DIR="/home/robot/droid"
DOCKER_COMPOSE_DIR="$ROOT_DIR/.docker/laptop"
DOCKER_COMPOSE_FILE="$DOCKER_COMPOSE_DIR/docker-compose-laptop.yaml"

PARAMETERS_FILE="/home/robot/droid/droid/misc/parameters.py"
awk -F'[[:space:]]*=[[:space:]]*' '/^[[:space:]]*([[:alnum:]_]+)[[:space:]]*=/ && $1 != "ARUCO_DICT" { gsub("\"", "", $2); print "export " $1 "=" $2 }' "$PARAMETERS_FILE" > temp_env_vars.sh
source temp_env_vars.sh
export ROOT_DIR=$ROOT_DIR
export NUC_IP=$nuc_ip
export LAPTOP_IP=$laptop_ip
export ROBOT_IP=$robot_ip
export SUDO_PASSWORD=$sudo_password
export ROBOT_TYPE=$robot_type
export ROBOT_SERIAL_NUMBER=$robot_serial_number
export HAND_CAMERA_ID=$hand_camera_id
export VARIED_CAMERA_1_ID=$varied_camera_1_id
export VARIED_CAMERA_2_ID=$varied_camera_2_id
export LIBFRANKA_VERSION=$libfranka_version
export ROOT_DIR=$ROOT_DIR
rm temp_env_vars.sh

if [ "$ROBOT_TYPE" == "panda" ]; then
        export LIBFRANKA_VERSION=0.9.0
else
        export LIBFRANKA_VERSION=0.10.0
fi

echo "Select an Ethernet interface to set a static IP for:"

interfaces=$(ip -o link show | grep -Eo '^[0-9]+: (en|eth|ens|eno|enp)[a-z0-9]*' | awk -F' ' '{print $2}')

# Display available interfaces for the user to choose from
if [ "${#interfaces[@]}" -eq 0 ]; then
    # No options found
    echo "No interfaces found, please esatablish one."
elif [ "${#interfaces[@]}" -eq 1 ]; then
    # Only one option found: auto-select it
    interface_name="${interfaces[0]}"
    echo "Only one interface found: $interface_name. Auto-selecting..."
else
    # Multiple options found: show the selection menu
    echo "Multiple interfaces found. Please choose one:"
    select interface_name in "${interfaces[@]}"; do
        if [ -n "$interface_name" ]; then
            break
        else
            echo "Invalid selection. Please choose a valid interface."
        fi
    done
fi

echo "You've selected: $interface_name"

# Add and configure the static IP connection
nmcli connection delete "laptop_static"
nmcli connection add con-name "laptop_static" ifname "$interface_name" type ethernet
nmcli connection modify "laptop_static" ipv4.method manual ipv4.address $LAPTOP_IP/24
nmcli connection up "laptop_static"

echo "Static IP configuration complete for interface $interface_name."

echo "Starting ADB server on host machine"
adb kill-server
adb -a nodaemon server start &> /dev/null &

echo -e "run client application \n"
docker compose -f $DOCKER_COMPOSE_FILE run --remove-orphans laptop_setup "python3 /app/scripts/main.py"

#!/usr/bin/env bash
# AWS port of deploy/gcp/bootstrap-ssd-raid.sh: RAID0 every instance-store NVMe device onto /data
# for high-throughput CSV generation (the dataset generator host only — Aerospike nodes use raw
# devices, see bootstrap-aerospike-node.sh).
#
# Instance-store devices are found by model ("Amazon EC2 NVMe Instance Storage"), so the EBS root
# volume is never touched. Instance store is wiped if the instance is stopped; /data is scratch.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then echo "Run as root: sudo bash $0"; exit 1; fi

export DEBIAN_FRONTEND=noninteractive
command -v mdadm >/dev/null || { apt-get update -qq && apt-get install -y -qq mdadm python3; }

mapfile -t DISKS < <(lsblk -dpno NAME,MODEL | awk '/Amazon EC2 NVMe Instance Storage/ {print $1}')
if [[ ${#DISKS[@]} -lt 1 ]]; then echo "No instance-store NVMe disks found"; lsblk; exit 1; fi
echo "Found instance-store disks: ${DISKS[*]}"

if ! mdadm --detail /dev/md0 &>/dev/null; then
  if [[ ${#DISKS[@]} -eq 1 ]]; then
    mkfs.ext4 -F -E nodiscard "${DISKS[0]}"; DEV="${DISKS[0]}"
  else
    echo yes | mdadm --create /dev/md0 --level=0 --raid-devices=${#DISKS[@]} "${DISKS[@]}"
    mkfs.ext4 -F -E nodiscard /dev/md0; DEV=/dev/md0
  fi
else
  DEV=/dev/md0
fi

mkdir -p /data
grep -q ' /data ' /etc/fstab || echo "${DEV} /data ext4 defaults,nofail,noatime 0 2" >> /etc/fstab
mount -a
chown -R "${SUDO_USER:-ubuntu}:$(id -gn "${SUDO_USER:-ubuntu}")" /data
df -h /data
echo "RAID0 ready at /data"

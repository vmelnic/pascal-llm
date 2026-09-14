#!/usr/bin/env bash

pascal_profile_value() {
  local profile_file="$1"
  local variable_name="$2"
  awk -v variable_name="${variable_name}" '
    index($0, variable_name "=") == 1 {
      print substr($0, length(variable_name) + 2)
      found = 1
      exit
    }
    END { if (!found) exit 1 }
  ' "${profile_file}"
}

pascal_validate_cuda_device_set() {
  local device_set="$1"
  [[ "${device_set}" =~ ^[0-9]+(,[0-9]+)*$ ]]
}

pascal_cuda_device_sets_overlap() {
  local left="$1"
  local right="$2"
  local left_device
  local right_device

  pascal_validate_cuda_device_set "${left}" || return 2
  pascal_validate_cuda_device_set "${right}" || return 2

  IFS=',' read -r -a left_devices <<< "${left}"
  IFS=',' read -r -a right_devices <<< "${right}"
  for left_device in "${left_devices[@]}"; do
    for right_device in "${right_devices[@]}"; do
      [[ "${left_device}" != "${right_device}" ]] || return 0
    done
  done
  return 1
}

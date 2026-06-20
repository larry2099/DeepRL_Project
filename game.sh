#!/usr/bin/env bash

# This script launches a Windows executable using Proton within a Linux environment.
# It requires exactly one argument: the path to the executable to run.
# It sets up a separate Proton prefix for each executable to avoid conflicts.
# Usage: proton <path-to-executable>

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <executable-path> [proton-path-suffix]"
    exit 1
fi

# determine where your proton binary is, the below assumes default Steam location.
# then update STEAM_ROOT and PROTON_VER

PROTON_ROOT="$HOME/.proton"
PROTON_VER="Proton - Experimental"
STEAM_ROOT="$HOME/.steam/root" # expects: steamapps/common/$PROTON_VER/proton
GAME_ROOT="$(dirname "$1")"
GAME="$(basename "$1")"
PROTON_PATH="$PROTON_ROOT/$GAME"
if [ "$#" -eq 2 ]; then
    PROTON_PATH="$PROTON_ROOT/$GAME-$2"
    cp -r "$PROTON_ROOT/$GAME" $PROTON_PATH
fi

cd "$GAME_ROOT" || exit

mkdir -p $PROTON_PATH
export STEAM_COMPAT_DATA_PATH=$PROTON_PATH
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_ROOT"

"$STEAM_ROOT/steamapps/common/$PROTON_VER/proton" run "$1" >/dev/null 2>&1

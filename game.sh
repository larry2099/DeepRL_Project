#!/usr/bin/env bash

GAME_EXE="$HOME/Games/GD/GeometryDash.exe"
SPEEDHACK="$(pwd)/bin/speedhack.so"

PROTON_ROOT="$HOME/.proton"
PROTON_VER="Proton - Experimental"
STEAM_ROOT="$HOME/.steam/root" # expects: steamapps/common/$PROTON_VER/proton
GAME_ROOT="$(dirname "$GAME_EXE")"
GAME="$(basename "$GAME_EXE")"
PROTON_PATH="$PROTON_ROOT/$GAME"
cd "$GAME_ROOT" || exit

mkdir -p $PROTON_PATH
export STEAM_COMPAT_DATA_PATH=$PROTON_PATH
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_ROOT"
# export LD_PRELOAD="$SPEEDHACK"

"$STEAM_ROOT/steamapps/common/$PROTON_VER/proton" run "$GAME_EXE" >/dev/null 2>&1 &

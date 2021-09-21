{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    zsh
    python39
    python39Packages.pip
    pipenv
  ]);
  runScript = ''
  zsh
  '';
}).env

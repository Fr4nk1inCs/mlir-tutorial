{
  description = "Devshell for MLIR development";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      mkLLVMShell = pkgs.mkShell.override {stdenv = pkgs.llvmPackages.stdenv;};
    in {
      devShell = mkLLVMShell rec {
        packages = with pkgs; [
          # build tools
          cmake
          ninja
          llvmPackages.bintools

          # development tools
          clang-tools
          cmake-format
          cmake-lint
          neocmakelsp
        ];

        buildInputs = with pkgs; [
          python3
          ncurses
          zlib
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
        '';
      };
    });
}

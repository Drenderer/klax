{
  description = "Klax Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils}:
    flake-utils.lib.eachDefaultSystem ( system:
      let
        inherit (nixpkgs) lib;
        pkgs = nixpkgs.legacyPackages.${system};

        # Update Python version if required
        python = pkgs.python313;
      in {


        # Development shell for installing klax using uv in editable mode
        devShells = {
          # This is a impure virtual environment workflow
          default = pkgs.mkShell {
            packages = with pkgs; [
              python
              uv
            ];

            env = 
              {
                # Prevent uv from managing python downloads
                UV_PYTHON_DOWNLOADS = "never";
                # Force ov to use nixpkgs Python interpreter
                UV_PYTHON = python.interpreter;
              }
              // lib.optionalAttrs pkgs.stdenv.isLinux {
                # Python libraries ofen load native shared object using dlopen(3).
                # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of 
                # libraries without using RPATH for lookup.
                LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
              };

            shellHook = ''
              unset PYTHONPATH
              source .venv/bin/activate
            '';
          };
        };
      }
  );
}

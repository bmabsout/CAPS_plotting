{
  description = "A very basic flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
          extensions = (with pkgs.vscode-extensions; [
            bbenoist.Nix
            ms-python.python
          ]);
          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscodium;
            vscodeExtensions = extensions;
          };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (ps: with ps; 
              [ numpy tqdm matplotlib scipy ])
            )
            vscodium-with-extensions
          ];
          Test=123;
        };
      }
    );
}

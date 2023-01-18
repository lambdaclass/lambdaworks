{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
   name = "cuda-env-shell";
   buildInputs = with pkgs; [
     git curl cargo rustc
   ];        
}

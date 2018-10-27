with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "dpar-env";
  env = buildEnv { name = name; paths = buildInputs; };

  nativeBuildInputs = [
    latest.rustChannels.stable.rust
    pkgconfig
    ragelDev
    latest.rustChannels.stable.rust
  ];

  buildInputs = [
    hdf5
    libtensorflow
    openssl
  ];
}

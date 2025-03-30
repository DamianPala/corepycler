# corepycler

-----

`corepycler` is a tool for testing CPU stability, especially under single-core load. It was inspired by [corecycler](https://github.com/sp00n/corecycler).
The main reason for developing this tool is that `corecycler` supports only Windows.
`corepycler` is written in Python and is intended to be cross-platform. Currently, it supports only Linux.

## Table of Contents

- [Installation](#installation)
- [License](#license)
- [Usage](#usage)

## Installation

```bash
sudo apt install pipx -y
pipx install hatch
git clone git@github.com:DamianPala/tig-manager.git && cd corepycler
hatch shell
corepycler
```

## Usage

### Configuration

Dump configuration file using:

```shell
corepycler -d
```

Parameter description is included in this config file that is YAML formated. You can use this file later by:

```shell
corepycler -c <path-to-config-file>
```

### Profiles

Use:

```shell
corepycler -h
```

to list avaliable profiles, that are handy to make different tests. Eg.

```shell
corepycler -p CO_STABILITY_FAST
```

## License

`corepycler` is distributed under the terms of the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

# GranularScenes

> Seeing scenes with multi-granularity!


## Setup and running

1. Clone. 
1. Run `./setup.sh cont_build python julia` to build the container and setup enviroment
2. Enter `./run.sh julia` to get into Julia REPL


> NOTE: when setting up Julia, if `Pkg` complains about some packages not being registered, you can install them manually via `Pkg.add(git url)`.

This project has automatic configuration!! This configuration is defined in `default.conf`.
You should always prepend `./run.sh` before any command (including running programs like `julia`) to ensure consistency. 
If you wish to have different values than `default.conf`, simply:

``` sh
cp default.conf user.conf
vi user.conf # edit to your liking without adding new elements
```

## Mac and Window users

In order to use singularity you must have a virtual machine running. 
Assuming you have vagrant (and something like virtualbox) setup on your host, you can follow these steps

### Using `setup.sh`


### Using `run.sh`

Provision the virtual machine defined in `Vagrantfile` with:

``` sh
vagrant up
```




Create a `user.conf` as described above

> Note: git will not track `user.conf`

Modify `user.conf` such that `path` is set to route through vagrant

``` yaml
[ENV]
path:vagrant ssh -c singularity
```


## Contributing

### Contributing rules

1. place all re-used code in packages (`src` or `pydeps`)
2. place all interactive code in `scripts`
3. not use "hard" paths. Instead update `PATHS` in the config.
4. add contributions to branches derived from `master` or `dev`
4. not use `git add *`
5. not commit large files (checkpoints, datasets, etc). Update `setup.sh` accordingly.


### Project layout

The python package environment is located under `pydeps` and can be imported using `import pydepts`

Likewise, the Julia package is described under `src` and `test`

All scripts are located under `scripts` and data/output is under `output` as specific in the project config (`default.conf` or `user.conf`)


### Changing the enviroment

To add new python or julia packages use the provided package managers. The Python dependencies are listed under `env.d/requirements.txt`

For julia you can also use `] add ` in the REPL

## Related works

> Manuscript pending!

### Conference submissions

Belledonne, M., Bao, Y., & Yildirim, I. (2022). Navigational affordances are automatically computed during scene perception: Evidence from behavioral change blindness and a computational model of active attention. *Journal of Vision, 22*(14), 4128-4128.

Belledonne, M., & Yildirim, I. (2021). Automatic computation of navigational affordances explains selective processing of geometry in scene perception: Behavioral and computational evidence. In *Proceedings of the Annual Meeting of the Cognitive Science Society* (Vol. 43, No. 43).

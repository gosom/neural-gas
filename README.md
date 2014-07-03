neural-gas
==========

Exercise for gameAI 2014 - Implementation of the Growing Neural Gas Algorithm

Dependencies
-----------

    numpy
    networkx
    pygame

Install the latest versions.

The app is tested on Fedora Linux 32bit.
Usage
-----

see the commandline arguments with:

`python ngas.py -h`

Mandatory is only the input file:

`python ngas.py simpleDungeonMap.txt`

To modify the parameters use commanline options:

    --tmax : default=10000)
    --delta_c :default=0.05
    --delta_n : default=0.0005
    --max_age : defaultn=25
    --lambda_ : default=100
    --nNodes : default=100)
    --alpha : default=0.5
    --beta : default=0.0005
    --slow : default=False

Use `--slow` if you want the display to be slower (`python ngas.py simpleDungeonMap.txt --slow`)

Credits
------
Dimitris & Giorgos


    



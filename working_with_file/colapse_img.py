import shutil
import os

//copy dataraw to one folder
src = '/home/lmtruong1512/Pictures/Data/animals'
dst = '/home/lmtruong1512/Pictures/Data/collapsed_animals'
with os.scandir(src) as animals:
    for species in animals:
        count = 0
        species_dir = os.path.join(src, species.name)
        with os.scandir(species_dir) as individuals:
            for individual in individuals:
                count += 1
                print(str(count) + " " + os.path.join(dst,
                                                      species.name + "_" + str(count)))
                shutil.copy(os.path.join(species_dir, individual.name),
                            os.path.join(dst, species.name + "_" + str(count)))

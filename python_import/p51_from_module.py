from machine.car import drive
from machine.tv import watch

drive()
watch()
print('=====================')

from machine import car, tv
car.drive()
tv.watch()
print('======================')

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

from machine import test
test.car.drive()
test.tv.watch()
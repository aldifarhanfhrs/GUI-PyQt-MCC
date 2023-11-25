class Hero:
    def __init__(self, x, y):
        self.name = x
        self.power = y
        


heroOne = Hero('Alucard', 100)
heroTwo = Hero('Lancelot', 1)

print(heroOne.name, heroOne.power)
print(heroOne.power)

"""heroOne.nama = 'Alucard'
heroOne.nyawa = 1000

heroTwo.nama = 'Lancelot'
heroTwo.nyawa = 3000

print(heroOne.__dict__)"""
import utils


if __name__ == '__main__':
    tmp = utils.load_csv('Reviews.csv', 'Text')

    for x, y, mask in utils.generator(tmp):
        for i in range(x.shape[0]):
            foo = utils.decode(x[i])
            bar = utils.decode(y[i])
            tmp = mask[i]

            print(foo)
            print(bar)
            print(tmp)
            print()
            import pdb; pdb.set_trace()

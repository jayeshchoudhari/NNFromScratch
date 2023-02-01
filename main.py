from src.nn import Neuron, Layer, MLP

if __name__ == '__main__':

    n = MLP(3, [4,4,1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]


    for k in range(20):
        #forward pass
        ypred = [n(x) for x in xs]
        loss = sum([(yground - yout)**2 for yground, yout in zip(ys, ypred)])

        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(k, loss.data)
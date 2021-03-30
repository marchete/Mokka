# Mokka
Mokka is a minimal Inference Engine for Dense Layer Neural Networks. Written on a single C++ header, it uses AVX2.
The code is aimed to give good performance on a minimal binary size, without external references.
Most Inference Engines are bloated with external libraries, complex loaders that inflates the binary size.

These engines are too big to use it in AI challenges (i.e. www.codingame.com ), were file size is limited to <160KB without external libraries.

Current Test binary output is like 30KB in size, including MNIST source code that can be removed when not used. It's feasible to have a compressed binary of 60KB + 80KB of weights.


## Training
Trust on proven frameworks. There is no point on reinvent the wheel and create your own framework to train your network.
Mokka is just an inference engine, you need to train somewhere else and export weights.
I use Tensorflow 2.0 to train some MNIST neural networks (even with GPU enabled).

Saving model was done with:
```python
def SaveModel(my_model,fileSTR):
    totalbytes=0
    data=[]
    Wmodel = open(fileSTR, "wb")
    for x in my_model.weights:
        nn = x.numpy()
        T = nn
        v = np.ndarray.tobytes(T)
        Wmodel.write(bytearray(v))
        totalbytes+=len(v)
        print(x.name, len(v)," dims:",nn.ndim," ", T.shape)
        data.append(base64.b64encode(v).decode("utf-8"))
    Wmodel.close()
    print("Total bytes:"+str(totalbytes))
```
You can see how the SaveModel works in the Jupyter notebooks. 

Model in Tensorflow must match the Model created on Mokka. For the sake of binary size there aren't validity checks when loading weights.

## Requirements
1. Tested on Ubuntu 18 (WSL)
2. C++ compiler, I used Clang++ 9. It can be compiled in Visual Studio too.
3. UPX for binary compression.
4. Tensorflow 2.0 for training, it can be even another PC. I have TF2.0 but on Windows, because I can use CUDA for GPU acceleration.
5. MNIST datasets. TF2 will download them automatically. but the test binary will need to download them separately. Use ```./DOWNLOAD_MNIST.sh``` to download, extract and rename the MNIST dataset. It will create a new ./mnist folder.

## Creating a model

It's inspired on Tensorflow. Create a layer, then feed that layer to the next one.
```c++
Model CreateModel(shared_ptr<Input>& input,shared_ptr<Layer>& policy,shared_ptr<Layer>& value ){
	shared_ptr<Layer> x,split;
	#define NN(M_M) make_shared<M_M>
	input	=  NN(Input)(vector<int>{28*28});
	x	=(*NN(Dense)("Dense", 128,RELU))(input);
	policy	=(*NN(Dense)("Soft", 10,SOFTMAX))(x);
	#undef NN
	Model model({input},{policy});
	return model;
}
```
Some structures are flattened to W\*H instead of being a {W,H} matrix

## MNIST test

I've tested the accuracy of the code with two MNIST tests. The code achieves 12 us/sample, that is a good performance. Tensorflow is faster, 9us/sample.
They are called ```MNIST Simple.ipynb``` and ```MNIST Simple29.ipynb```, they are Jupyter Notebooks. If you run both notebooks they will create two weight files on ./Mokka subfolder, called ```DENSE.weights``` and ```DENSE29.weights``` respectively.

Running the Test (requires clang++9):

```bash
cd Mokka
./DOWNLOAD_MNIST.sh
./COMPILE.sh Test_MNIST
./Test_MNIST
```

When you run the binary file you'll get some accuracy percentages, these % are the same than on Jupyter notebooks.

**Testing MNIST Simple29.ipynb**

![Test2a](https://github.com/marchete/Mokka/raw/main/img/Test2a.JPG)

![Test2b](https://github.com/marchete/Mokka/raw/main/img/Test2b.JPG)

Accuracy is the same than on Test

![Test2c](https://github.com/marchete/Mokka/raw/main/img/Test2c.JPG)

Similar summary. Same number of trainable parameters

**Testing MNIST Simple.ipynb**

![Test1a](https://github.com/marchete/Mokka/raw/main/img/Test1a.JPG)

![Test1b](https://github.com/marchete/Mokka/raw/main/img/Test1b.JPG)

Accuracy is the same than on Test

![Test1c](https://github.com/marchete/Mokka/raw/main/img/Test1c.JPG)

Similar summary. Same number of trainable parameters

**Binary size**

![BinarySize](https://github.com/marchete/Mokka/raw/main/img/CompileSize.JPG)**

30KB, including MNIST load code

Tester will also save a ```DENSE.test``` file. These are an export of the loaded weights, the file should be exactly the same as ```DENSE.weights```.

## Future work

1. Convolutional Network Layers. Convolutional are expensive to run and hard to implement, more in AVX2. They should be in optional ```#define```, to only use them when needed.
2. AlphaZero implementation.

## References

Mokka was based on Latte:
https://github.com/amrmorsey/Latte

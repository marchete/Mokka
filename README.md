# Mokka
Mokka is a minimal Inference Engine for Dense and Convolutional 2D Layer Neural Networks. Written on a single C++ header, it uses AVX2.
The code is aimed to give good performance on a minimal binary size, without external references. At some special cases (one-hot inputs with more zeros than ones, or ReLU outputs) it's 200% faster than Tensorflow.
Most Inference Engines are bloated with external libraries, complex loaders that inflates the binary size.

These engines are too big to use it in AI challenges (i.e. www.codingame.com ), were file size is limited to <160KB without external libraries.

Current Test binary output is like 32KB in size, including MNIST source code that can be removed when not used. It's feasible to have a compressed binary of 40KB + 110KB of weights (Succesfully tested on CGZero, an AlphaZero like bot that uses Mokka ).


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

I've tested the accuracy of the code with three MNIST tests. The code achieves 4 us/sample, that is a good performance. Tensorflow needs 9us/sample. That's because I optimize according to inputs. Zero value inputs are skipped, and 1.0 inputs adds without multiplying the input.
They are called ```MNIST Simple.ipynb``` , ```MNIST Simple29.ipynb``` and  ```MNIST CNN.ipynb```, they are Jupyter Notebooks. If you run both notebooks they will create three weight files on ./Mokka subfolder, called ```DENSE.weights``` , ```DENSE29.weights``` and ```CNN_T.weights``` respectively.

Running the Test (requires clang++9):

```bash
cd Mokka
./DOWNLOAD_MNIST.sh
./COMPILE.sh Test_MNIST
./Test_MNIST
```

When you run the binary file you'll get some accuracy percentages, these % are the same than on Jupyter notebooks.

**Testing MNIST Simple29.ipynb**

![Test2a](https://github.com/marchete/Mokka/raw/main/img/Test2a_.JPG)

![Test2b](https://github.com/marchete/Mokka/raw/main/img/Test2b.JPG)

Accuracy is the same than on Test

![Test2c](https://github.com/marchete/Mokka/raw/main/img/Test2c.JPG)

Similar summary. Same number of trainable parameters

**Testing MNIST Simple.ipynb**

![Test1a](https://github.com/marchete/Mokka/raw/main/img/Test1a_.JPG)

![Test1b](https://github.com/marchete/Mokka/raw/main/img/Test1b.JPG)

Accuracy is the same than on Test

![Test1c](https://github.com/marchete/Mokka/raw/main/img/Test1c.JPG)

Similar summary. Same number of trainable parameters

**Testing MNIST CNN.ipynb**

![Test3a](https://github.com/marchete/Mokka/raw/main/img/Test3a.JPG)

![Test3b](https://github.com/marchete/Mokka/raw/main/img/Test3b.JPG)

Accuracy is the same than on Test

![Test3c](https://github.com/marchete/Mokka/raw/main/img/Test3c_.JPG)

Similar summary. Same number of trainable parameters.

Unfortunately I found Convolutional inference too slow. It's 15x slower than Dense. I need to recheck if I'm duplicating some work or something, but the output is the same expected value. I profiled the code on Visual Studio, 78% of the time is spent on the function `WeightBiasLayer::calculateOutput(Tensor &input_mat)`
on line:
```
output.xmm[i].v = _mm256_fmadd_ps(fm, weights[N].xmm[i].v, output.xmm[i].v);
```

**Binary size**

![BinarySize](https://github.com/marchete/Mokka/raw/main/img/CompileSize.JPG)**

32KB, including MNIST load code

Tester will also save a ```DENSE.test``` file. These are an export of the loaded weights, the file should be exactly the same as ```DENSE.weights```.

**Weights size**

I've added two functions to Compress weights file from float32 to float16. That means you can compress 50% the weights file. 

The float32->float16->float32 doesn't degrade a lot the accuracy. I've used that feature on Oware's CGZero and the bot is performant.

```file32to16(string f32, string f16)```

```file16to32(string f16, string f32)```

## Future work

1. Custom kernel for Convolutional layers, to allow hex grid inputs. I already added it but needs testing. Kernel filter should be like:
 ```
  0 1 1
  1 1 1
  1 1 0
  ```
3. ~~Convolutional Network Layers. Convolutional are expensive to run and hard to implement, more in AVX2. They should be in optional ```#define```, to only use them when needed.~~ Implemented a Convolutional 2D layer. It has padding, stride, but it seems slow. I tried to cache it and transform it as a Dense-like calculation. Even with AVX2 it's dead slow. 71us/sample vs 4us/sample from pure Dense tests.
4. ~~AlphaZero implementation.~~ A similar Alphazero NN was succesfully implemented, see https://github.com/marchete/CGZero

## References

Mokka was based on Latte:
https://github.com/amrmorsey/Latte

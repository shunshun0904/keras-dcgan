## KERAS-DCGAN

Implementation of http://arxiv.org/abs/1511.06434 with the (awesome) [keras](https://github.com/fchollet/keras) library, for generating artificial images with deep learning.

This trains two adversarial deep learning models on real images, in order to produce artificial images that look real.

The generator model tries to produce images that look real and get a high score from the discriminator.

The discriminator model tries to tell apart between real images and artificial images from the generator.

---

This assumes theano ordering.
You can still use this with tensorflow, by setting "image_dim_ordering": "th" in ~/.keras/keras.json (although this will be slower).

---

## Usage

**Training:**

 `python dcgan.py --mode train --batch_size <batch_size>`

ex

`python dcgan.py --mode train --path ~/images --batch_size 128`

**Image generation:**

`python dcgan.py --mode generate --batch_size <batch_size>`

`python dcgan.py --mode generate --batch_size <batch_size> --nice` : top 5% images according to discriminator

python dcgan.py --mode generate --batch_size 128

---

## Result

**generated images :**

![generated_image.png](./assets/generated_image.png)

![nice_generated_image.png](./assets/nice_generated_image.png)

**train process :**

![training_process.gif](./assets/training_process.gif)

---

## 事前知識

※DIC受講生の場合、Keras入門を参照

### keras

`keras document`
https://keras.io/ja/

## import

`Dence`

https://keras.io/ja/layers/core/

通常の全結合ニューラルネットワークレイヤー


`Reshape`

https://keras.io/ja/layers/core/

>あるshapeに出力を変形する．

```py
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (Non
```

`Activation`

```
引数
activation： 使用する活性化関数名 (activationsを参照)， もしくは，TheanoかTensorFlowオペレーション．
```

`Flatten`

https://keras.io/ja/layers/core/#flatten

```
keras.layers.core.Flatten()
入力を平滑化する．バッチサイズに影響されない．
```

```py
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

`BatchNormalization`

[from: Batch normalization layer (Ioffe and Szegedy, 2014)](https://arxiv.org/abs/1502.03167)

各バッチ毎に前の層の出力（このレイヤーへの入力）を正規化します． つまり，平均を0，標準偏差値を1に近づける変換を適用します．

https://keras.io/ja/layers/normalization/

`UpSampling2D`

https://keras.io/ja/layers/convolutional/#upsampling2d

```
データの行と列をそれぞれsize[0]及びsize[1]回繰り返します．
```

![](https://zo7.github.io/img/2016-09-25-generating-faces/deconv.png)

>https://zo7.github.io/blog/2016/09/25/generating-faces.html

`Conv2D`

https://keras.io/ja/layers/convolutional/#conv2d

```
2次元入力をフィルターする畳み込み層．

use_biasをTrueにすると，バイアスベクトルが出力に加えられます．activationがNoneでない場合，指定した活性化関数が出力に適用されます．

このレイヤーをモデルの第1層に使うときはキーワード引数input_shape （整数のタプル，サンプル軸を含まない）を指定してください． 例えば，data_format="channels_last"のとき，128x128 RGB画像ではinput_shape=(128, 128, 3)となります．
```

`MaxPooling2D`

空間データのマックスプーリング演算．

https://keras.io/ja/layers/pooling/#maxpooling2d

`SGD`

 https://keras.io/ja/optimizers/#sgd

```
モーメンタム，学習率減衰，Nesterov momentumをサポートした確率的勾配降下法．
```
`PIL`

https://pillow.readthedocs.io/en/4.3.x/

```
Python の画像処理ライブラリで、Python Imaging Library (PIL)の fork プロジェクト。非常に高速にチューニングされており、同様なライブラリであるImageMagickよりも常に高速に動作する。getpixel/putpixelは非常に低速なため、画像生成以外の目的使用する場合は他を使用するのがおすすめらしい。
```

`argparseモジュール`
http://tech.uribou.tokyo/python-argparsenoshi-ifang/

```
コマンドラインツールの開発に必要。
```

## DCGANについて
- GANは具体的なネットワークの構成に言及していない。（少なくとも論文中では）
- DCGAN(Deep Convolutional Generative Adversarial Networks) は、GANに対して畳み込みニューラルネットワークを適用して、うまく学習が成立するベストプラクティスについて提案したもの。
- 元になった論文 Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks リンク
要点をまとめると下記のようになる
- プーリング層を全て下記で置き換える
    - D(鑑別器): 畳み込み層（strided convolutions）(これはいわゆる普通の畳み込み層のことらしい)
    - G(生成器): 分数的にストライドする畳み込み層 (fractional-strieded convolutions)(これはすなわちdeconvolution a.k.a. transposed convolutionsのことらしい...)
- バッチノルムを使用する（重要らしい）
- 深い構成では全結合層を除去する
- 生成器ではReLUを出力層以外の全ての層で活性化関数として使用し、出力層ではtanhを使用する
- 識別器ではLeakyReLUを全ての層で使用する

## generator_model

![](https://elix-tech.github.io/images/2017/gan/dcgan_generator.png)

[Radford et al. (2015)](https://arxiv.org/abs/1511.06434)より引用

```py
def generator_model():

    #kerasのシーケンシャルモデルの定義
    model = Sequential()
   
    #全結合レイヤーを作成
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
   
 　　　　　　
    model.add(Dense(128*7*7))
    
    # バッチ正規化
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    
    # アップサンプリング
    model.add(UpSampling2D(size=(2, 2)))
    
    # 畳み込み層
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    
    # アップサンプリング
    model.add(UpSampling2D(size=(2, 2)))
    
    # 畳み込み層
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model
```
http://yusuke-ujitoko.hatenablog.com/entry/2017/05/08/010314　から引用
> もともとBatchNormalizationを入れていなかった。 何度試しても、異なるノイズをもとに生成しているにも関わらず、同じ一様な画像となってしまうという問題が発生しており、 層がそこそこ深いためか、勾配がうまく伝わっていないと思われたため、 BatchNormalizationを加えて、各層の平均を0に、分散を正規化した。 これにより、異なるノイズからは少なくとも異なる画像が生成されるようになった。

## discriminator_model

<a href="https://diveintocode.gyazo.com/3a5fa4581e5a5187d0f342cd1babe645"><img src="https://t.gyazo.com/teams/diveintocode/3a5fa4581e5a5187d0f342cd1babe645.png" alt="https://diveintocode.gyazo.com/3a5fa4581e5a5187d0f342cd1babe645" width="1612"/></a>
> https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0 から引用

```py
def discriminator_model():
    
    # モデルの定義
    model = Sequential()
    
    # 畳み込み層の作成
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    
    # プーリング層の作成
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 畳み込み層の作成
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    
    # プーリング層の作成
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    # 全結合層の作成
   　　model.add(Dense(1024))
    model.add(Activation('tanh'))
    
    # 全結合層の作成
   　　model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
```

## generator_containing_discriminatoring
```py
def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model
```    
ジェネレータとディスクリミネータを繋いだモデル。誤差伝搬時に使う。

KerasでネットワークのWeightを固定させて、別のLayerのみ学習させたい時に、trainableを使用する。これをfreezeという。
- 以下trainable適用の思考実験.ネット上でも結構議論されているので、使い方が難しそう。trainable→compileの流れ
https://qiita.com/mokemokechicken/items/937a82cfdc31e9a6ca12
https://qiita.com/obsproth/items/d7c53580b847fe762da7
http://www.mathgram.xyz/entry/keras/tips/freeze
https://qiita.com/t-ae/items/236457c29ba85a7579d5 (compileを学習のたびに切り替える必要があるかどうかの議論)


出力画像を一つの画像にまとめて保存する関数。
```py
def combine_images(generated_images):
    num = generated_images.shape[0]  # shape[0]はgenerated_imagesの配列のながさ
    width = int(math.sqrt(num))　　　　　　　　　　　　# 画像を正方形とした場合の一辺のながさ
   　　height = int(math.ceil(float(num)/width))　　　　　　　　　　　　　　　　　　　　　　　　# ()内の切り上げ
    shape = generated_images.shape[1:3]　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　# shape配列のslice
    image = np.zeros((height*shape[0], width*shape[1]),  
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)           
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \ 
            img[:, :, 0]
    return image
```    


```py
def train(BATCH_SIZE):  
    #mnistデータを取得。
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #画像を正規化してX_trainに入れ直す。
    X_train = (X_train.astype(np.float32) - 127.5)/127.5　# # RGBのカラービット数で正規化（0〜255）
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    d = discriminator_model()
    g = generator_model()
    #ジェネレータとディスクリミネータと２つを結合したモデルを定義。
    d_on_g = generator_containing_discriminator(g, d)
    #ジェネレータとディスクリミネータと２つを結合したモデル用の最適化関数をSGDで定義。
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # trainable更新後にcompileしないと反映されない
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    # trainable更新後にcompileしないと反映されない
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            # (128,100)のサイズのノイズを作成。
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            # ????
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # ノイズをジェネレータに入力。
            generated_images = g.predict(noise, verbose=0)
            # なぜ20で割っているのか？
            if index % 20 == 0:
                image = combine_images(generated_images)
                # 正規化を戻している
                image = image*127.5+127.5
                # 画像の出力形式の指定
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            # 元画像と出力した画像を結合してXとする。        
            X = np.concatenate((image_batch, generated_images))
            # バッチサイズ文の（0と1）の配列作成
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            #ディスクリミネータにXとyを入力し学習し誤差を出す。
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            # [-1:1]の範囲で乱数の生成
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            # 層の重みの更新をしないように設定
            d.trainable = False
            #２つのモデルを結合したモデルの学習をし誤差をだす。
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            # 層の重みを更新する設定
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            # index = 90 の時のみ実行??
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)

```

kerasの使用上、インスタンス化直後はすべてtrainableな状態になるので、compileする必要がある。trainableの更新を反映させるために、compileする


生成部分の定義。学習時にsave_weightsしてるので、load_weightsする。
niceはデフォルトで実行するとFalse。niceを指定すると良い推定値の画像がソートされて纏めて保存される。

```py
# 画像生成部分の定義
def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    # インスタンス化直後はtrainableになるので、compile
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    # train関数での重みをロード
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        # インスタンス化直後はtrainableになるので、compile
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        # train関数での重みをロード
        d.load_weights('discriminator')
        # (バッチサイズ*20,100)のサイズの乱数を生成
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        # 予測モデルにより画像生成
        generated_images = g.predict(noise, verbose=1)
        # 生成画像をディスクリミねーたーに判定させる
        d_pret = d.predict(generated_images, verbose=1)
        # (0,バッチサイズ*20)のnumpy配列の作成
        index = np.arange(0, BATCH_SIZE*20)
        # 上の配列を転置している（行と列を逆にしている）
        index.resize((BATCH_SIZE*20, 1))
        # 
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")
```



## 実行時の引数作成
```py
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args
```    

## ファイルを実行するmain関数
```py
if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
```        

## 畳み込みアニメーション

`from`
https://github.com/vdumoulin/conv_arithmetic#convolution-animations

<table style="width:100%">
  <tr>
    <td><img src="gif/no_padding_no_strides.gif"></td>
    <td><img src="gif/arbitrary_padding_no_strides.gif"></td>
    <td><img src="gif/same_padding_no_strides.gif"></td>
    <td><img src="gif/full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
    <td>Full padding, no strides</td>
  </tr>
  <tr>
    <td><img src="gif/no_padding_no_strides_transposed.gif"></td>
    <td><img src="gif/arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img src="gif/same_padding_no_strides_transposed.gif"></td>
    <td><img src="gif/full_padding_no_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Half padding, no strides, transposed</td>
    <td>Full padding, no strides, transposed</td>
  </tr>
  <tr>
    <td><img src="gif/no_padding_strides.gif"></td>
    <td><img src="gif/padding_strides.gif"></td>
    <td><img src="gif/padding_strides_odd.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
    <td></td>
  </tr>
  <tr>
    <td><img src="gif/no_padding_strides_transposed.gif"></td>
    <td><img src="gif/padding_strides_transposed.gif"></td>
    <td><img src="gif/padding_strides_odd_transposed.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides, transposed</td>
    <td>Padding, strides, transposed</td>
    <td>Padding, strides, transposed (odd)</td>
    <td></td>
  </tr>
  <tr>
    <td><img src="gif/dilation.gif"></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, no stride, dilation</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>



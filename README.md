# CSP

A pytorch implementation of CSP, `Wei Liu, etc. High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection, CVPR 2019.` 

The authors' Keras implementation [liuwei16/CSP](https://github.com/liuwei16/CSP).

still working in progress...

## Updates

- 20 March 2020, Load keras weights into pytorch model, and got the same model outputs as `resnet50 nn_p3p4p5` in [liuwei16/CSP/resnet50.py](https://github.com/liuwei16/CSP/blob/785bc4c5f956860116d8d51754fd76202afe4bcb/keras_csp/resnet50.py#L264)
- 23 March 2020, finish test_caltech, got 4.06 MR under Reasonable setting using authors trained weights.

## Environments

- ubuntu16.04
- cuda10.0
- python3.6
- pytorch1.4.0

## Test One Picture

Download trained weights from authors' baidu netdisk [BaiduYun](https://pan.baidu.com/s/1SSPQnbDP6zf9xf8eCDi3Fw) (Code: jcgd).

For this test, we are using ResNet-50 initialized from CityPersons: Height+Offset prediciton: model_CSP/caltech/fromcity/net_e82_l0.00850005054218.hdf5

We'll use this keras hdf5 weights, and load it in pytorch, and run inference.

```
git clone https://github.com/wang-xinyu/csp.pytorch
cd csp.pytorch
// put net_e82_l0.00850005054218.hdf5 here
python test_one_pic.py
```

The out.jpg as follows will be generated.

<p align="center">
<img src="https://user-images.githubusercontent.com/15235574/77291176-abf7ae00-6d18-11ea-9c11-dd65c7d69507.jpg">
</p>

## Eval Caltech

We still use net_e82_l0.00850005054218.hdf5 above.

Put caltech test images(4024 images) in `data/caltech/images/`. The dir tree is as below.

```
data
    caltech
        test.txt
        images
            set06_V000_I00029.jpg
            set06_V000_I00059.jpg
            ...
            set10_V011_I01709.jpg
```

Get `AS.mat` from [liuwei16/CSP](https://github.com/liuwei16/CSP/blob/master/eval_caltech/AS.mat), put into `eval_caltech/`

Then run test_caltech.py to generate results, and use octave to run eval and get MR.

```
python test_caltech.py
cd eval_caltech/
sudo apt install octave
octave-cli
   -> dbEval
```

A result file `eval_caltech/ResultsEval/eval-newReasonable.txt` will generated.



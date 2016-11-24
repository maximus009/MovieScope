# MovieScope
Idea is to extend video classification for identifying what the genre (romance, horror or action) of a movie is based
on the trailer solely using visual features.
<br>
Currently, using only spatial CNN features (VGG), to extract features for every frame extracted.
<br>
Frame extraction can be a challenge in itself. Currently, sampling frame at a fixed timestep. Can also implement
shot-boundary detection, and split the trailers into scenes which might (or not) be useful in training.

<i>MovieScope</i> is certainly, started with movie trailers, which are long videos compared to other videos used for
activity classification and recognition. Will need to use recurrent networks and LSTMs as well Refer to branch "lstm" to
see the implementation.

<br>

To run the code, first, make changes to the file <code>config/resources.py</code> with the appropriate directory.
<br>
|
data<br>
|___train<br>
    |___<class1><br>
        |___<video1><br>
        |___<video2><br>
        . <br>
        . <br>
        . <br>
    |___<class2> <br>
        |___<video1> 
        |_<video2>
        .
        .
        .
    .
    .
    .
|___test
    |___<class1>
        |___<video1>
        |___<video2>
        .
        .
        .
    |___<class2>
        |___<video1>
        |_<video2>
        .
        .
        .
    .
    .
    .

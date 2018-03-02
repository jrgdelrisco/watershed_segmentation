# watershed_segmentation

Some implementations of the __Watershed__ algorithm for image segmentation.

The code was written in _C++11_, using some of its new features, and using _OpenCV_ for the image processing.

There is a command line application called _watershed.exe_ for testing the algorithms. Type _watershed -h_ for help. This command line application uses the [Lean Mean Option Parser](http://optionparser.sourceforge.net/optionparser.h).

## An example
One of the most interesting implementations of the Watershed algorithm is called [Watershed with viscous force](http://users.cecs.anu.edu.au/~sgould/papers/accv12-watershed.pdf). This algorithm's modification makes it more tolerant to noise and images with incomplete borders. Use one of the given images (_noise.tif_ or _openborders.tif_) to check out this:

    >watershed.exe -i -v
    >enter image file name: path/to/noise.tif

Then draw a marker outside the object and another marker inside. Press 's' to segment the image and observe the results.

You can compare results using another Watershed implementation.
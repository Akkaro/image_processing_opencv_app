/*  in the frequency domain, the process of convolution simplifies to multiplication => faster than in the spatial domain
    the output is simply given by F(u,v) x G(u,v) where F(u,v) and G(u,v) are the Fourier transforms of their respective functions
    The frequency-domain representation of a signal carries information about the signal's magnitude and phase at each frequency*/

/*
The algorithm for filtering in the frequency domain is:
    a) Perform the image centering transform on the original image (9.15)
    b) Perform the DFT transform
    c) Alter the Fourier coefficients according to the required filtering
    d) Perform the IDFT transform
    e) Perform the image centering transform again (this undoes the first centering transform)
 */

void centering_transform(Mat img){
//expects floating point image
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
        }
    }
}

Mat generic_frequency_domain_filter(Mat src)
{

    // Discrete Fourier Transform: https://docs.opencv.org/4.2.0/d8/d01/tutorial_discrete_fourier_transform.html
    int height = src.rows;
    int width = src.cols;

    Mat srcf;
    src.convertTo(srcf, CV_32FC1);
    // Centering transformation
    centering_transform(srcf);

    //perform forward transform with complex image output
    Mat fourier;
    dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

    // the frequency is represented by its real and imaginary parts called frequency coefficients
    // split into real and imaginary channels fourier(i, j) = Re(i, j) + i * Im(i, j)
    Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
    split(fourier, channels);  // channels[0] = Re (real part), channels[1] = Im (imaginary part)

    //calculate magnitude and phase of the frequency by transforming it from cartesian to polar coordinates
    // the magnitude is useful for visualization

    Mat mag, phi;
    magnitude(channels[0], channels[1], mag); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6d3b097586bca4409873d64a90fe64c3
    phase(channels[0], channels[1], phi); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9db9ca9b4d81c3bde5677b8f64dc0137


    // TODO: Display here the log of magnitude (Add 1 to the magnitude to avoid log(0)) (see image 9.4e))
    // do not forget to normalize (you can use the normalize function from OpenCV)

    // TODO: Insert filtering operations here (channels[0] = Re(DFT(I), channels[1] = Im(DFT(I)); low pass or high pass filters
    // low pass filters equation 9.16 and equation 9.17
    // high pass filters equation 9.18 and 9.19


    //perform inverse transform and put results in dstf
    Mat dst, dstf;
    merge(channels, 2, fourier);
    dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT);

    // Inverse Centering transformation
    centering_transform(dstf);

    //normalize the result and put in the destination image
    normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

    return dst;
}
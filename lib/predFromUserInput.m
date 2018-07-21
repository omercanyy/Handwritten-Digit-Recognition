function p = predFromUserInput(h)
    clc; clear all; close;
    set(0, 'DefaultFigureWindowStyle','docked');
    
    h = imread('try.jpg');
    hNTSC = rgb2ntsc(h);
    hBW = hNTSC(:,:,1);
    minVal = min(hBW(:));
    maxVal = max(hBW(:));
    diff = maxVal - minVal;
    hNorm = (hBW - minVal)/diff;
    %imshow(hNorm);
    %pause;
	%h = double(h)/255;
    imshow(hNorm);
    load('lib\Trained_NN.mat');
    p = mod(predict(Theta1, Theta2, hNorm(:)'), 10);
    fprintf('The digit you draw is %d\n.', p);
end
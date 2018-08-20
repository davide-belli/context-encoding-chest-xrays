% Generate 'diff' patch as pixel-wise difference of the real and reconstructed
% patches

folder = './healthy ok/';
imnames = strcat(folder,'*_real.png');
imagefiles = dir(imnames); 
count = length(imagefiles);


for i=1:count
   realname = imagefiles(i).name;
   imagename = realname(1:14);
   reconname = strcat(imagename, '_recon.png');
   a = imread(strcat(folder, realname));
   b = imread(strcat(folder, reconname));
   
   c = abs(a-b);
   c = c*2; % double the brightness value to make low differences more evident
   imshow(c);
   pause(0.2);
   
   diffname = strcat(imagename, '_diff.png');
   imwrite(c, strcat(folder, diffname));
   
end
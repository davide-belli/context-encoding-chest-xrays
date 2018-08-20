% Computes Structural Similarity Index and Mean-squared error.
% This scores are used in addition to PSNR to evaluate the reconstruction quality

healthy = 0;

if healthy == 1
    folder = './healthy880patch/';
    lenn = 14;
end
if healthy == 0
    folder = './patches/';
    lenn = 12;
end

imnames = strcat(folder,'*_real.png');
imagefiles = dir(imnames); 
count = length(imagefiles);
tot_mse = [];
tot_sim = [];
tot_psnr = [];


for i=1:count
   realname = imagefiles(i).name;
   imagename = realname(1:lenn);
   reconname = strcat(imagename, '_recon.png');
   a = imread(strcat(folder, realname));
   b = imread(strcat(folder, reconname));
   a = a(33:96, 33:96);
   b = b(33:96, 33:96);
  
   mse = immse(a, b);
   [sim, ~] = ssim(a, b);
   this_psnr = psnr(a, b);
   
   tot_mse = [tot_mse, mse];
   tot_sim = [tot_sim, sim];
   tot_psnr = [tot_psnr, this_psnr];
   
   diffname = strcat(imagename, '_diff.png');
   
end

avg_mse = mean(tot_mse);
avg_sim = mean(tot_sim);
avg_psnr = mean(tot_psnr);

std_mse = std(tot_mse);
std_sim = std(tot_sim);
std_psnr = std(tot_psnr);

disp([avg_mse, std_mse]);
disp([avg_sim, std_sim]);
disp([avg_psnr, std_psnr]);

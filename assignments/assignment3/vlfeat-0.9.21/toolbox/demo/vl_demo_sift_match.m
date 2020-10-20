% VL_DEMO_SIFT_MATCH  Demo: SIFT: basic matching

pfx = fullfile(vl_root,'figures','demo') ;
randn('state',0) ;
rand('state',0) ;
figure(1) ; clf ;

% --------------------------------------------------------------------
%                                                    Create image pair
% --------------------------------------------------------------------

Ia = imread(fullfile(vl_root,'data','roofs1.jpg')) ;
Ib = imread(fullfile(vl_root,'data','roofs2.jpg')) ;

% --------------------------------------------------------------------
%                                           Extract features and match
% --------------------------------------------------------------------

[fa,da] = vl_sift(im2single(rgb2gray(Ia))) ;
[fb,db] = vl_sift(im2single(rgb2gray(Ib))) ;

[matches, scores] = vl_ubcmatch(da,db) ;

[drop, perm] = sort(scores, 'descend') ;
matches = matches(:, perm) ;
scores  = scores(perm) ;

figure(1) ; clf ;
imagesc(cat(2, Ia, Ib)) ;

xa = fa(1,matches(1,:)) ;
xb = fb(1,matches(2,:)) + size(Ia,2) ;
ya = fa(2,matches(1,:)) ;
yb = fb(2,matches(2,:)) ;

hold on ;
h = line([xa ; xb], [ya ; yb]) ;
set(h,'linewidth', 1, 'color', 'b') ;

axis image off ;


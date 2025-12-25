function xss=calc_ss(A,C,nx,ny,ys)
[V,D]=eig(A);
V2=V(:,nx-ny+1:end);
zi=(C*V2)\ys;
xss=V2*zi;
function [Atil,Btil,Ctil] = immpc(A,B,C)
% immpc calculates state-space model in incremental form from positional state-space
% model for MPC controllers
nx = size(A,1);
nu = size(B,2);
ny = size(C,1);
Atil = [A B; zeros(nu,nx) eye(nu)];
Btil = [B ; eye(nu)];
Ctil = [C zeros(ny,nu)];

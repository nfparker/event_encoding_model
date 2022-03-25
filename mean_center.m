function[answer]=mean_center(matrix)

mean_mat=mean(matrix);
matrix_new=zeros(size(matrix));

for rowz=1:numel(matrix(1,:))
   matrix_new(:,rowz)=matrix(:,rowz)-mean_mat(rowz);
end

answer=matrix_new;
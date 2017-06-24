% 
% Tests to check that slimmed-down dcm code and demos return 
% same outputs as full functions


% Compare original and slimmed model reduction erp demos

[Y_orig] = test_DEMO_model_reduction_ERP;

[Y_slim] = slim_DEMO_model_reduction_ERP;

figure; hold on;
plot(Y_orig.y{1}, 'b')
plot(Y_slim.y{1}, 'r')



% Compare two runs of original test demo

[Y_orig_1] = test_DEMO_model_reduction_ERP;

[Y_orig_2] = test_DEMO_model_reduction_ERP;

figure; hold on;
plot(Y_orig_1.y{1}, 'b')
plot(Y_orig_2.y{1}, 'r')





function [ ] = export2csv( res )
ids = 1:size(res,2);
csvwrite_with_headers('submission.csv', [ids',res'], {'Id', 'Prediction'});


end


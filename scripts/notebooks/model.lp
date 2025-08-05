\ Model fractional_lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.9177238883352644 y[0] + 0.55093332756855 y[1] + 0.4726297551640858 y[2]
   + 0.9851344714759253 y[3] + 0.8809746613726694 y[4]
   + 0.3812824759920631 y[5] + 0.0658224693132493 y[6]
   + 0.7945895650618318 y[7] + 0.6525254552409978 y[8]
   + 0.4843759374331108 y[9] + 0.9354988475787182 y[10]
   + 0.0125364284098161 y[11] + 0.4795312226630113 y[12]
   + 0.7834618165819082 y[13] + 0.5783200117889661 y[14]
   + 0.4144811839083593 y[15]
Subject To
 normalize: y[0] + y[1] + y[2] + y[3] + y[4] + y[5] + y[6] + y[7] + y[8]
   + y[9] + y[10] + y[11] + y[12] + y[13] + y[14] + y[15] = 1
 upper_bound_y_0: y[0] - t <= 0
 upper_bound_y_1: y[1] - t <= 0
 upper_bound_y_2: y[2] - t <= 0
 upper_bound_y_3: y[3] - t <= 0
 upper_bound_y_4: y[4] - t <= 0
 upper_bound_y_5: y[5] - t <= 0
 upper_bound_y_6: y[6] - t <= 0
 upper_bound_y_7: y[7] - t <= 0
 upper_bound_y_8: y[8] - t <= 0
 upper_bound_y_9: y[9] - t <= 0
 upper_bound_y_10: y[10] - t <= 0
 upper_bound_y_11: y[11] - t <= 0
 upper_bound_y_12: y[12] - t <= 0
 upper_bound_y_13: y[13] - t <= 0
 upper_bound_y_14: y[14] - t <= 0
 upper_bound_y_15: y[15] - t <= 0
 coverage_0_(1,_array([_1,__2,__5,__6,__9,_10,_13,_14])): y[1] + y[2]
   + y[5] + y[6] + y[9] + y[10] + y[13] + y[14] - t >= 0
 coverage_1_(1,_array([_5,__6,_13,_14])): y[5] + y[6] + y[13] + y[14] - t
   >= 0
 coverage_2_(1,_array([_9,_10])): y[9] + y[10] - t >= 0
 coverage_3_(1,_array([1,_2])): y[1] + y[2] - t >= 0
 coverage_4_(1,_array([_5,__6,_13,_14])): y[5] + y[6] + y[13] + y[14] - t
   >= 0
 coverage_5_(1,_array([_1,__2,__5,__6,__9,_10,_13,_14])): y[1] + y[2]
   + y[5] + y[6] + y[9] + y[10] + y[13] + y[14] - t >= 0
 coverage_6_(1,_array([_5,__6,__9,_10,_13,_14])): y[5] + y[6] + y[9]
   + y[10] + y[13] + y[14] - t >= 0
 coverage_7_(1,_array([_1,__2,__5,__6,_13,_14])): y[1] + y[2] + y[5] + y[6]
   + y[13] + y[14] - t >= 0
 coverage_8_(1,_array([],_dtype=int64)): - t >= 0
 coverage_9_(1,_array([_1,__2,__9,_10])): y[1] + y[2] + y[9] + y[10] - t
   >= 0
 coverage_10_(1,_array([_1,__2,__5,__6,_13,_14])): y[1] + y[2] + y[5]
   + y[6] + y[13] + y[14] - t >= 0
 coverage_11_(1,_array([],_dtype=int64)): - t >= 0
 coverage_12_(1,_array([_5,__6,__9,_10,_13,_14])): y[5] + y[6] + y[9]
   + y[10] + y[13] + y[14] - t >= 0
 coverage_13_(1,_array([_5,__6,_13,_14])): y[5] + y[6] + y[13] + y[14] - t
   >= 0
 coverage_14_(1,_array([_9,_10])): y[9] + y[10] - t >= 0
 coverage_15_(1,_array([_1,__2,__9,_10])): y[1] + y[2] + y[9] + y[10] - t
   >= 0
 coverage_16_(1,_array([_5,__6,__9,_10,_13,_14])): y[5] + y[6] + y[9]
   + y[10] + y[13] + y[14] - t >= 0
 coverage_17_(1,_array([],_dtype=int64)): - t >= 0
 coverage_18_(1,_array([_9,_10])): y[9] + y[10] - t >= 0
 coverage_19_(1,_array([_5,__6,_13,_14])): y[5] + y[6] + y[13] + y[14] - t
   >= 0
 coverage_20_(1,_array([_9,_10])): y[9] + y[10] - t >= 0
 coverage_21_(1,_array([1,_2])): y[1] + y[2] - t >= 0
 coverage_22_(1,_array([],_dtype=int64)): - t >= 0
 coverage_23_(1,_array([_5,__6,__9,_10,_13,_14])): y[5] + y[6] + y[9]
   + y[10] + y[13] + y[14] - t >= 0
Bounds
End

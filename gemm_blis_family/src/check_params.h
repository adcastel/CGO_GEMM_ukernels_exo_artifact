  {
    int info = 0, notA, notB, nrowA, ncolA, nrowB, ncolB;

    notA = (transA=='N');
    notB = (transB=='N');
    if (notA) {
      nrowA = m;
      ncolA = k;
    }
    else {
      nrowA = k;
      ncolA = m;
    }
    if (notB) {
      nrowB = k;
      if (orderB=='R')
        ncolB = n;
    }
    else {
      nrowB = n;
      if (orderB=='R')
        ncolB = k;
    }
  
    if ( (orderA!='C')&&(orderA!='R') )
      info = 1; 
    if ( (orderB!='C')&&(orderB!='R') )
      info = 2; 
    if ( (orderC!='C')&&(orderC!='R') )
      info = 3; 
    if ( (!notA)&&(!(transA=='T')) )
      info = 4; 
    else if ( (!notB)&&(!(transB=='T')) )
      info = 5;
    else if ( m<0 )
      info = 6;
    else if ( n<0 )
      info = 7;
    else if ( k<0 )
      info = 8;
    else if ( (orderA=='C')&&(ldA<max(1, nrowA)) )
      info = 11;
    else if ( (orderA=='R')&&(ldA<max(1, ncolA)) )
      info = 11;
    else if ( (orderB=='C')&&(ldB<max(1, nrowB)) )
      info = 13;
    else if ( (orderB=='R')&&(ldB<max(1, ncolB)) )
      info = 13;
    else if ( (orderC=='C')&&(ldC<max(1, m)) ) 
      info = 16;
    else if ( (orderC=='R')&&(ldC<max(1, n)) )
      info = 16;
      
    if ( info!=0 ) {
      printf("Error: incorrect parameter to gemm_blis %d \n", info);
      exit(-1);
    }
  }

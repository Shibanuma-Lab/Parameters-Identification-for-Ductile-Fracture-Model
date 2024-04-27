      SUBROUTINE UEXTERNALDB(LOP,LRESTART,TIME,DTIME,KSTEP,KINC)
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION TIME(2)
#include <SMAAspUserSubroutines.hdr>
	  INTEGER JFFLAG_new
      INTEGER K1, K2,I
      CHARACTER*200 OUTDIR
      CHARACTER*200 try_out 
	  INTEGER IFTHERE 
	  pointer(ptr_JFFLAG_new,JFFLAG_new(200000,8))
	  pointer(ptr_IFTHERE,IFTHERE(200000))
	  
	  IF(lop.EQ.0) THEN
      call MutexInit(1)
      call MutexLock(1)
 	  ptr_JFFLAG_new = SMAIntArrayCreate(1,200000*8,0)
 	  ptr_IFTHERE = SMAIntArrayCreate(2,200000,0)
	  IF(get_thread_id().eq. 0) THEN
         CALL GETOUTDIR( OUTDIR, LENOUTDIR )
         WRITE(try_out,'(A,A)')TRIM(OUTDIR), '\umat_path.csv'
	     OPEN(80,file=try_out, status='replace')
	  	 WRITE(80,'(A,A)') 'UMAT1, TRY1,UMAT2, TRY2,F,FT'
         CLOSE(80)
      ENDIF
	  call MutexUnlock(1)
	  END IF
      RETURN
      END SUBROUTINE UEXTERNALDB


      SUBROUTINE UMAT(STRESS, STATEV, DDSDDE, SSE, SPD, SCD, RPL,
     1 DDSDDT, DRPLDE, DRPLDT, STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP,
     2 PREDEF, DPRED, CMNAME, NDI, NSHR, NTENS, NSTATV, PROPS, NPROPS,
     3 COORDS, DROT, PNEWDT, CELENT, DFGRD0, DFGRD1, NOEL, NPT, LAYER,
     4 KSPT, KSTEP, KINC)
C
      INCLUDE 'ABA_PARAM.INC'
#include <SMAAspUserSubroutines.hdr>
      CHARACTER*8 CMNAME
      CHARACTER*100 FILENAME
      CHARACTER*100 FILENAME1
      CHARACTER*100 OUTDIR
      LOGICAL THERE
	  CHARACTER*200 try_out 
	  INTEGER JFFLAG_new
	  INTEGER IFTHERE 
 	  pointer(ptr_JFFLAG_new,JFFLAG_new(200000,8))
 	  pointer(ptr_IFTHERE,IFTHERE(200000))
	  
C
C ID - Indicator of stage
C      0: Elastic stage
C      1: Hardening stage
C      2: Softening stage
C      3: Fracture stage
C ID3- Indicater of stage
C      0: Other stage
C      1: Fracture stage
C
C EELAS - Elastic strain tensor
C EPLAS - Plastic strain tensor
C FLOW  - Direction of plastic flow
C
C PROPS(1)   - Young乫s modulus
C PROPS(2)   - Poisson乫s ratio
C PROPS(3)   - Trixiality vs critical plastic strain parameter 1
C PROPS(4)   - Trixiality vs critical plastic strain parameter 2
C PROPS(5..) - Mises hardening data
C
C NHARD - Number of hardening data
C SLOPE  - Slope of softening curve
C SRMIN  - Ratio of minimum stress to maximum stress
C SRTRN  - Ratio of transition stress to maximum stress
C NSOFT   - Number of softening data
C NED - Number of element delete judge
C
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 PREDEF(1),DPRED(1),PROPS(NPROPS), COORDS(3),DROT(3, 3),
     3 DFGRD0(3,3),DFGRD1(3,3),EELAS(6),EPLAS(6),FLOW(6),HARD(3),
     4 SOFT(2,2000),DEVSTRESS(6),JFFLAG(8)
C
      PARAMETER(ZERO=0.D0, ONE=1.D0, TWO=2.D0, THREE=3.D0, SIX=6.D0,
     1 ENUMAX=.4999D0, NEWTON=10, TOLER=1.0D-6)
C
      NHARD=3000
      SLOPE=-5000.0D0
      SRMIN=0.02D0
      SRTRN=0.019D0
      NSOFT=100
      NED=1
	  MINTRA= ONE/THREE

C
C---------------------------------------------------------------------
C
C Variables from previous increment
C
      CALL ROTSIG(STATEV(1),DROT,EELAS,2,NDI,NSHR)
      CALL ROTSIG(STATEV(NTENS+1),DROT,EPLAS,2,NDI,NSHR)
C
      ptr_JFFLAG_new = SMAIntArrayAccess(1)
	  ptr_IFTHERE = SMAIntArrayAccess(2)
      EQPLAS=STATEV(1+2*NTENS)
      DAMAGE=STATEV(2+2*NTENS)
      SYIELD=STATEV(3+2*NTENS)
      FFLAG=1
      SYIELDMAX=STATEV(3*NTENS)
      ID=STATEV(3+3*NTENS)
      ID3=STATEV(1+4*NTENS)
      SEQPLAS=STATEV(5+3*NTENS)
      IFFLAG=STATEV(4*NTENS)
      JSLOPE=STATEV(2+4*NTENS)
C
C Elastic properties
C
      EMOD=PROPS(1)
      ENU=MIN(PROPS(2), ENUMAX)
      EBULK3=EMOD/(ONE-TWO*ENU)
      EG2=EMOD/(ONE+ENU)
      EG=EG2/TWO
      EG3=THREE*EG
      ELAM=(EBULK3-EG2)/THREE
C
C Elastic stiffness
C
      DO K1=1, NDI
      DO K2=1, NDI
      DDSDDE(K2, K1)=ELAM
      END DO
      DDSDDE(K1, K1)=EG2+ELAM
      END DO
      DO K1=NDI+1, NTENS
      DDSDDE(K1, K1)=EG
      END DO
C
C Predictor stress and elastic strain tensors
C
      DO K1=1, NTENS
      DO K2=1, NTENS
      STRESS(K2)=STRESS(K2)+DDSDDE(K2, K1)*DSTRAN(K1)
      END DO
      EELAS(K1)=EELAS(K1)+DSTRAN(K1)
      END DO
C
C Equivalent Von Mises stress
C
      SMISES=(STRESS(1)-STRESS(2))**TWO+(STRESS(2)-STRESS(3))**TWO
     1 +(STRESS(3)-STRESS(1))**TWO
      DO K1=NDI+1,NTENS
      SMISES=SMISES+SIX*STRESS(K1)**TWO
      END DO
      SMISES=SQRT(SMISES/TWO)
	  SHYDRO=(STRESS(1)+STRESS(2)+STRESS(3))/THREE
      DO K1=1, NDI
      DEVSTRESS(K1)=STRESS(K1)-SHYDRO
      END DO
      DO K1=NDI+1,NTENS
      DEVSTRESS(K1)=STRESS(K1)
      END DO
C
C Flow directions
C
      DO K1=1,NDI
      FLOW(K1)=(STRESS(K1)-SHYDRO)/SMISES
      END DO
      DO K1=NDI+1, NTENS
      FLOW(K1)=STRESS(K1)/SMISES
      END DO
C
C---------------------------------------------------------------------
C 1 - Elastic stage
C---------------------------------------------------------------------
C
      ID=0
      IFFLAG=1
C
      TRIX=SHYDRO/SMISES
	  J2=((DEVSTRESS(1)-DEVSTRESS(2))**TWO+(DEVSTRESS(1)-DEVSTRESS(3))**TWO+(DEVSTRESS(3)-DEVSTRESS(2))**TWO)/6
     1 +DEVSTRESS(4)**TWO+DEVSTRESS(5)**TWO+DEVSTRESS(6)**TWO	  
      J3=DEVSTRESS(1)*DEVSTRESS(2)*DEVSTRESS(3)+TWO*DEVSTRESS(4)*DEVSTRESS(5)
     1 *DEVSTRESS(6)-DEVSTRESS(1)*(DEVSTRESS(6))**TWO-DEVSTRESS(2)*
     2 (DEVSTRESS(5))**TWO-DEVSTRESS(3)*(DEVSTRESS(4))**TWO
      IF (J2.EQ.ZERO) THEN
	    J2=STATEV(3+4*NTENS)
	    J3=STATEV(4+4*NTENS)
      ENDIF

	  ALODEpara=THREE*(THREE**(ONE/TWO))*J3/(TWO*J2**(THREE/TWO))
      IF (ALODEpara.LE.-ONE ) THEN
	    ALODEpara=-ONE
      ENDIF	  
      IF (ALODEpara.GE.ONE ) THEN
	    ALODEpara=ONE
      ENDIF	  	  
	  ALODE=(ONE/THREE)*ACOS(ALODEpara)
	  ALODEnorma=ONE-TWO*THREE*ALODE/ACOS(-ONE)


C Yield stress (hardening)
C
      IF(DAMAGE.LT.ONE) THEN
C
      SYIELD=ZERO
      HARD(1)=ZERO
C
      DO K1=1,NHARD
      EQPL1=PROPS(2*K1+6)
      IF(EQPLAS.LT.EQPL1) THEN
      EQPL0=PROPS(2*K1+4)
      DEQPL1=EQPL1-EQPL0
      SYIEL00=PROPS(2*K1+3)
      SYIEL10=PROPS(2*K1+5)
      DSYIEL=SYIEL10-SYIEL00
      HARD(1)=DSYIEL/DEQPL1
      SYIEL0=SYIEL00+(EQPLAS-EQPL0)*HARD(1)
      GOTO 10
      ENDIF
      END DO
   10 CONTINUE
C
C---------------------------------------------------------------------
C 2 - Hardening stage
C---------------------------------------------------------------------
C
      IF(SMISES.GT.(ONE+TOLER)*SYIEL0) THEN
      ID=1
C
C Yield stress and eq. plastic strain using Newton iteration
C
      SYIELD=SYIEL0
      DEQPL=ZERO
C
      DO KEWTON=1,NEWTON
      RHS=SMISES-EG3*DEQPL-SYIELD
      DEQPL=DEQPL+RHS/(EG3+HARD(1))
C
      SYIELD=ZERO
      HARD(1)=ZERO
      EQPLAS=EQPLAS+DEQPL
C
      DO K1=1,NHARD
      EQPL1=PROPS(2*K1+6)
      IF(EQPLAS.LT.EQPL1) THEN
      EQPL0=PROPS(2*K1+4)
      DEQPL1=EQPL1-EQPL0
      SYIEL00=PROPS(2*K1+3)
      SYIEL10=PROPS(2*K1+5)
      DSYIEL=SYIEL10-SYIEL00
      HARD(1)=DSYIEL/DEQPL1
      SYIELD=SYIEL00+(EQPLAS-EQPL0)*HARD(1)
      GOTO 11
      ENDIF
      END DO
C
   11 CONTINUE
      EQPLAS=EQPLAS-DEQPL
      IF(ABS(RHS).LE.TOLER*SYIEL0) GOTO 12
C
      END DO
   12 CONTINUE
C
      ENDIF
C
C---------------------------------------------------------------------
C Common part for 3-Softening and 4-Fractured stages
C---------------------------------------------------------------------
C
      ELSE
      IFFLAG=0
C
C---------------------------------------------------------------------
C 3 - Softening stage
C---------------------------------------------------------------------
C
      IF(ID3.NE.1) THEN
C
      IF(JSLOPE.EQ.0) THEN
      SYIELDMAX=SYIELD
      SEQPLAS=EQPLAS
      JSLOPE=1
      ENDIF
C
C Softening curve with transition
C
      IF(SRTRN.GT.SRMIN) THEN
C
      NS=NSOFT
C
      E1=SEQPLAS
      S1=SYIELDMAX
      S2=SRTRN*SYIELDMAX
      S3=SRMIN*SYIELDMAX
      E2=E1+(S2-S1)/SLOPE
C
      SOFT(1,1)=0.99D0*E1
      SOFT(2,1)=S1-0.01D0*SLOPE
      SOFT(1,2)=E1
      SOFT(2,2)=S1
      SOFT(1,3)=E2
      SOFT(2,3)=S2
C
      E3=E2-TWO*(S2-S3)/SLOPE
      B=(SLOPE/TWO)/(E2-E3)
      DO K1=4,NS-1
      SOFT(1,K1)=E2+(K1-2)/(NS-3)*(E3-E2)
      SOFT(2,K1)=S3+B*(SOFT(1,K1)-E3)**TWO
      ENDDO
C
      SOFT(1,NS)=100
      SOFT(2,NS)=S3
C
C Softening curve without transition (only linear part)
C
      ELSE
C
      NS=4
C
      E1=SEQPLAS
      S1=SYIELDMAX
      S3=SRMIN*SYIELDMAX
      E3=E1+(S3-S1)/SLOPE
C
      SOFT(1,1)=0.99D0*E1
      SOFT(2,1)=S1-0.01D0*SLOPE
      SOFT(1,2)=E1
      SOFT(2,2)=S1
      SOFT(1,3)=E3
      SOFT(2,3)=S3
C
      SOFT(1,4)=100
      SOFT(2,4)=S3
C
      ENDIF
C
      SYIELD=ZERO
      HARD(1)=ZERO
C 
      DO K1=1,NS-1
      EQPL1=SOFT(1,K1+1)
      IF(EQPLAS.LE.EQPL1) THEN
      EQPL0=SOFT(1,K1)
      DEQPL1=EQPL1-EQPL0
      SYIEL00=SOFT(2,K1)
      SYIEL10=SOFT(2,K1+1)
      DSYIEL=SYIEL10-SYIEL00
      HARD(1)=DSYIEL/DEQPL1
      SYIEL0=SYIEL00+(EQPLAS-EQPL0)*HARD(1)
      GOTO 13
      ENDIF
      END DO
   13 CONTINUE
C
C YIELD stress (softening)
C
      IF(SMISES.GT.(ONE+TOLER)*SYIEL0) THEN
      ID=2
C
C Yield stress and eq. plastic strain using Newton iteration
C
      SYIELD=SYIEL0
      DEQPL=ZERO
C
      DO KEWTON=1,NEWTON
      RHS=SMISES-EG3*DEQPL-SYIELD
      DEQPL=DEQPL+RHS/(EG3+HARD(1))
C 
      SYIELD=ZERO
      HARD(1)=ZERO
      EQPLAS=EQPLAS+DEQPL
C
      DO K1=1,NS
      EQPL1=SOFT(1,K1+1)
      IF(EQPLAS.LT.EQPL1) THEN
      EQPL0=SOFT(1,K1)
      DEQPL1=EQPL1-EQPL0
      SYIEL00=SOFT(2,K1)
      SYIEL10=SOFT(2,K1+1)
      DSYIEL=SYIEL10-SYIEL00
      HARD(1)=DSYIEL/DEQPL1
      SYIELD=SYIEL00+(EQPLAS-EQPL0)*HARD(1)
      GOTO 14
      ENDIF
      END DO
C
   14 CONTINUE
      EQPLAS=EQPLAS-DEQPL
      IF(ABS(RHS).LE.TOLER*SYIEL0) GOTO 15
C
      END DO
   15 CONTINUE
C
      IF(SYIELD.LE.SRMIN*SYIELDMAX*1.001D0) THEN
      ID3=1
      ID=3
      ENDIF
C
      ENDIF
C
C---------------------------------------------------------------------
C 4 - Fractured stage
C---------------------------------------------------------------------
C
      ELSE
C
C Minimum stress (fracture)
C
      SYIEL0=SRMIN*SYIELDMAX
C
      IF(SMISES.GT.(ONE+TOLER)*SYIEL0) THEN
      ID=3
C
C Yield stress and eq. plastic strain using Newton iteration
C
      SYIELD=SYIEL0
      DEQPL=ZERO
      HARD(1)=ZERO
C
      DO KEWTON=1,NEWTON
      RHS=SMISES-EG3*DEQPL-SYIELD
      DEQPL=DEQPL+RHS/EG3
      IF(ABS(RHS).LE.TOLER*SYIELD) GOTO 25
      END DO
C
   25 CONTINUE
C
      ENDIF
C
      ENDIF
C
      ENDIF
C---------------------------------------------------------------------
C Common part
C---------------------------------------------------------------------
      IF(ID.GE.1) THEN
C
C Stress and strain update
C
      DO K1=1,NDI
      STRESS(K1)=FLOW(K1)*SYIELD+SHYDRO
      EPLAS(K1)=EPLAS(K1)+THREE/TWO*FLOW(K1)*DEQPL
      EELAS(K1)=EELAS(K1)-THREE/TWO*FLOW(K1)*DEQPL
      END DO
C
      DO K1=NDI+1,NTENS
      STRESS(K1)=FLOW(K1)*SYIELD
      EPLAS(K1)=EPLAS(K1)+THREE*FLOW(K1)*DEQPL
      EELAS(K1)=EELAS(K1)-THREE*FLOW(K1)*DEQPL
      END DO
C
      EQPLAS=EQPLAS+DEQPL
      SHYDRO=(STRESS(1)+STRESS(2)+STRESS(3))/THREE
      DO K1=1, NDI
      DEVSTRESS(K1)=STRESS(K1)-SHYDRO
      END DO
      DO K1=NDI+1,NTENS
      DEVSTRESS(K1)=STRESS(K1)
      END DO
C
C Damage based on trixiality (and Lode angle parameter)
C
      TRIX=SHYDRO/SYIELD
	  J2=((DEVSTRESS(1)-DEVSTRESS(2))**TWO+(DEVSTRESS(1)-DEVSTRESS(3))**TWO+(DEVSTRESS(3)-DEVSTRESS(2))**TWO)/6
     1 +DEVSTRESS(4)**TWO+DEVSTRESS(5)**TWO+DEVSTRESS(6)**TWO	  
      J3=DEVSTRESS(1)*DEVSTRESS(2)*DEVSTRESS(3)+TWO*DEVSTRESS(4)*DEVSTRESS(5)
     1 *DEVSTRESS(6)-DEVSTRESS(1)*(DEVSTRESS(6))**TWO-DEVSTRESS(2)*
     2 (DEVSTRESS(5))**TWO-DEVSTRESS(3)*(DEVSTRESS(4))**TWO
	  
      IF (J2.EQ.ZERO) THEN
	    J2=STATEV(3+4*NTENS)
	    J3=STATEV(4+4*NTENS)
      ENDIF

	  ALODEpara=THREE*(THREE**(ONE/TWO))*J3/(TWO*J2**(THREE/TWO))
      IF (ALODEpara.LE.-ONE ) THEN
	    ALODEpara=-ONE
      ENDIF	  
      IF (ALODEpara.GE.ONE ) THEN
	    ALODEpara=ONE
      ENDIF	  	  
	  ALODE=(ONE/THREE)*ACOS(ALODEpara)
	  ALODEnorma=ONE-TWO*THREE*ALODE/ACOS(-ONE)

      IF (TRIX.LE.MINTRA) THEN
	 CRIEQPLAS=PROPS(3)*EXP(-PROPS(4)*MINTRA)
      ELSE 
	 CRIEQPLAS=PROPS(3)*EXP(-PROPS(4)*TRIX)
      ENDIF

C      CRIEQPLAS=((PROPS(3)-PROPS(4))*ALODE*ALODE+PROPS(4))*EXP(-2.08204*TRIX)
C
      DDAMAGE=DEQPL/CRIEQPLAS
      DAMAGE=DAMAGE+DDAMAGE
      IF(DAMAGE.GE.ONE) THEN
      DAMAGE=ONE
      IFFLAG=0
      ENDIF
C
C Jacobian (Material tangent)
C
      EFFG=EG*SYIELD/SMISES
      EFFG2=TWO*EFFG
      EFFG3=THREE/TWO*EFFG2
      EFFLAM=(EBULK3-EFFG2)/THREE
      EFFHRD=EG3*HARD(1)/(EG3+HARD(1))-EFFG3
C
      DO K1=1, NDI
      DO K2=1, NDI
      DDSDDE(K2, K1)=EFFLAM
      END DO
      DDSDDE(K1, K1)=EFFG2+EFFLAM
      END DO
C
      DO K1=NDI+1, NTENS
      DDSDDE(K1, K1)=EFFG
      END DO
C
      DO K1=1, NTENS
      DO K2=1, NTENS
      DDSDDE(K2, K1)=DDSDDE(K2, K1)+EFFHRD*FLOW(K2)*FLOW(K1)
      END DO
      END DO
C
      ENDIF
C
C State variables
C
      ID2=10+NOEL*4
C
      CALL GETOUTDIR( OUTDIR, LENOUTDIR )
C

 	  IF (IFTHERE(NOEL).EQ.1) THEN  
		  IFFLAGSUM=0
		  DO K1=1,8
		     IF (NPT.EQ.K1)JFFLAG_new(NOEL,K1)=IFFLAG
		     IFFLAGSUM=IFFLAGSUM+JFFLAG_new(NOEL,K1)
		  END DO
		  IF (IFFLAGSUM.LE.8-NED) FFLAG=0		  
      ELSE	  
		  IF (IFFLAG.EQ.0)THEN	  
		  DO K1=1,8
		     JFFLAG_new(NOEL,K1)=1
		  END DO
		  JFFLAG_new(NOEL,NPT)=IFFLAG
		  IFTHERE(NOEL)=1
		  ENDIF
      ENDIF



      DO K1=1, NTENS
      STATEV(K1)=EELAS(K1)
      STATEV(K1+NTENS)=EPLAS(K1)
      END DO
C
      STATEV(1+2*NTENS)=EQPLAS
      STATEV(2+2*NTENS)=DAMAGE
      STATEV(3+2*NTENS)=SYIELD
      STATEV(4+2*NTENS)=FFLAG
      STATEV(5+2*NTENS)=DEQPL
      STATEV(3*NTENS)=SYIELDMAX
      STATEV(1+3*NTENS)=TRIX
      STATEV(2+3*NTENS)=ALODE
      STATEV(3+3*NTENS)=ID
      STATEV(4+3*NTENS)=CRIEQPLAS
      STATEV(5+3*NTENS)=SEQPLAS
      STATEV(4*NTENS)=IFFLAG
      STATEV(1+4*NTENS)=ID3
      STATEV(2+4*NTENS)=JSLOPE
	  
      STATEV(3+4*NTENS)=J2
      STATEV(4+4*NTENS)=J3
      STATEV(5+4*NTENS)=ALODEpara
      STATEV(6+4*NTENS)=ALODE
      STATEV(7+4*NTENS)=ALODEnorma  
C
      RETURN
      END

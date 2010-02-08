OPENCV=/usr/local
FFMPEG=/usr/local
CFLAGS=-O3 -I$(OPENCV)/include/opencv -L$(FFMPEG)/lib -L$(OPENCV)/lib -I ../NeovisionII/ -I ../inc -I ../ 
CLIBS=-lhighgui -lcv -lcxcore -lcvaux -lml -lm -ljpeg -lpng -ltiff -lavformat -lavutil -lavcodec
OBJS_SERVER=image.o filter.o operations.o cbcl_model_internal.o rule.o svm_model.o cbcl_label_server.o nv2_common.o nv2_label_server.o
OBJS_MODEL=image.o filter.o operations.o cbcl_model_internal.o rule.o svm_model.o cbcl_test.o 
OBJS_MODEL2=image.o filter.o operations.o cbcl_model_internal.o rule.o svm_model.o cbcl_test_matlab.o 
CC=g++

TARGETS= cbcl_model cbcl_model_matlab
default:
	make $(TARGETS)

cbcl_model: $(OBJS_MODEL)
	$(CC) $(CFLAGS) -o cbcl_model $(OBJS_MODEL) $(CLIBS)
cbcl_model_matlab: $(OBJS_MODEL2)
	$(CC) $(CFLAGS) -o cbcl_model_matlab $(OBJS_MODEL2) $(CLIBS)
cbcl_label_server: $(OBJS_SERVER)
	$(CC) $(CFLAGS) -o cbcl_label_server $(OBJS_SERVER) $(CLIBS)
image.o : image.cpp 
	$(CC) -c $(CFLAGS) image.cpp
filter.o: filter.cpp
	$(CC) -c $(CFLAGS) filter.cpp
operations.o: operations.cpp 
	$(CC) -c $(CFLAGS) operations.cpp
cbcl_model_internal.o: cbcl_model_internal.cpp
	$(CC) -c $(CFLAGS) cbcl_model_internal.cpp
rule.o: rule.cpp
	$(CC) -c $(CFLAGS) rule.cpp
svm_model.o: svm_model.cpp
	$(CC) -c $(CFLAGS) svm_model.cpp
cbcl_test.o: cbcl_test.cpp
	$(CC) -c $(CFLAGS) cbcl_test.cpp
cbcl_test_matlab.o: cbcl_test_matlab.cpp
	$(CC) -c $(CFLAGS) cbcl_test_matlab.cpp
cbcl_label_server.o: cbcl_label_server.cpp
	$(CC) -c $(CFLAGS) cbcl_label_server.cpp
nv2_common.o: ../NeovisionII/nv2_common.c
	$(CC) -c $(CFLAGS) ../NeovisionII/nv2_common.c 
nv2_label_server.o:	../NeovisionII/nv2_label_server.c 
	$(CC) -c $(CFLAGS) ../NeovisionII/nv2_label_server.c 
clean: 
	rm *.o

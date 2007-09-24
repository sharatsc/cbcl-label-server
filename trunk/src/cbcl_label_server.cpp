#include "nv2_common.h"
#include "nv2_label_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "image.h"
#include "filter.h"
#include "operations.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include "cbcl_model_internal.h"
#include "opts.h"
#include <string>
#include <vector>
#include <deque>
#include "svm_model.h"
#include "nv2_common.h"
#include <getopt.h>


using namespace std;

typedef struct __tag_out
{
public:
  double score; 
  string lbl;
}output_t;

class compare_outputs:binary_function<output_t,output_t,bool>
{
public:
  bool operator()(const output_t& lhs,const output_t& rhs)
  {
    return lhs.score < rhs.score;
  }
};

void load_filter(const char* filename,vector<filter>& filt)
{
  int ncount;
  ifstream fin;
  fin.open(filename,ifstream::in);
  fin>>ncount;
  filt.clear();filt.resize(ncount);
  for(int i=0;i<ncount;i++)
    fin>>filt[i];
  fin.close();
}

void init_opts(model_options* opt)
{
  int start_stop[]={0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7};
  int space_sum[] ={8,10,12,14,16,18,20,22};
  opt->nbands     = 8;
  opt->nscales    = 8;
  opt->ndirs      = 4;
  for(int i=0;i<opt->nbands;i++)
    {
      opt->scale_sum.push_back(start_stop[2*i]);
      opt->scale_sum.push_back(start_stop[2*i+1]);
      opt->space_sum.push_back(space_sum[i]);
    }
} 

image buf2img(const unsigned char* buf,int ht,int wt)
{
   image res(ht,wt);
   for(int i=0;i<ht;i++)
   {
     for(int j=0;j<wt;j++)
       res[i][j] = (double)buf[i*wt+j]/255;
   } 
   return res;
}

int main(int argc,char* argv[])
{
  using namespace std;
  vector<filter>          fb;
  vector<filter>          patches;
  vector<image>           s1;
  vector<image>           c1;
  vector<image>           s2;
  vector<double>          c2;
  vector<double>          scores;
  vector<output_t>        out;
  typedef vector<double>  vec_double_t;
  deque<vec_double_t>     qout;
  svm_model               model;
  compare_outputs         comp_func;
  /*command line options*/
  string id                   = "CBCL";
  int    in_port              = NV2_PATCH_READER_PORT;
  string patch_server         = "127.0.0.1";
  int    out_port             = NV2_LABEL_READER_PORT;
  int    interval             = 1;
  double threshold            = 0;
  int    memory               = 0;

  /*get overriding options*/
  static struct option long_options[]= {
          {"in-port",1,0,0},
          {"out-port",1,0,0},
          {"patch-server",1,0,0},
          {"interval",1,0,0},
          {"memory",1,0,0},
          {"threshold",1,0,0},
          {0,0,0,0}
  };
  int     param;
  int     long_opt_index;

  if(argc<2)
    {
      printf("Usage is %s --in-port=<listen> --out-port=<send> --patch-server=<server name> --interval=<skip_interval> --memory=<accumulate len> --threshold=<background threshold>\n",argv[0]);
      return 1;
    }
 /*program arguments*/
  while((param= getopt_long(argc,argv,"m:t:",long_options,&long_opt_index))!=-1)
  {
          switch(long_opt_index)
          {
          case 0:
                  in_port    = atoi(optarg);
                  break;
          case 1:
                  out_port   = atoi(optarg);
                  break;
          case 2:
                  patch_server=optarg;
                  break;
          case 3:
                  interval    = atoi(optarg);
                  break;
          case 4:
                  memory      = atoi(optarg);
                  break;
          case 5: 
                  threshold   = atof(optarg);
                  break;
          }
  }
  /*dump options*/
  cout<<"Identity    : "<<id<<endl;
  cout<<"Patch Server: "<<patch_server<<endl;
  cout<<"IN  port    : "<<in_port<<endl;
  cout<<"OUT port    : "<<out_port<<endl;
  cout<<"Memory      : "<<memory<<endl;
  cout<<"Threshold   : "<<threshold<<endl;
  
  image img; 
  model_options opt;
  init_opts(&opt);
  load_filter("gabor_bank.txt",fb);
  load_filter("patches.txt",patches);
  load_model("svm_model.txt",model);
  /*initialize output vector*/
  const int NCLASS = model.labels.size();
  out.resize(NCLASS);
  for(int j=0;j<NCLASS;j++)
  {
     out[j].lbl   = model.labels[j];
  }

  /*create server*/
  struct nv2_label_server* s =    nv2_label_server_create(in_port,
                                  patch_server.c_str(),
                                  out_port);
  /*process the patches*/
  while (1)
  {
        struct     nv2_image_patch p;
        const enum nv2_image_patch_result res = nv2_label_server_get_current_patch(s, &p);
        if (res == NV2_IMAGE_PATCH_END)
        {
           fprintf(stdout, "ok, quitting\n");
           break;
        }
        else if (res == NV2_IMAGE_PATCH_NONE)
        {
          usleep(10000);
		  fprintf(stdout, ".");
          continue;
        }
        //---------------------------------------------------
        //classify the image
        //
        //----------------------------------------------------
        img = buf2img(p.data,p.height,p.width);
		//imwrite(img,"img-01.jpg");
		img=imresize(img,128,(float)p.width/p.height*128);
		s1_baseline(img,fb,opt,s1);
        c1_baseline(s1,opt,c1);
        c2_baseline(c1,s2,patches,opt,c2);
        model.classify(c2,scores);
        scores.push_back(threshold);
        //---------------------------------------------------
        //process the scores
        //---------------------------------------------------
        qout.push_back(scores);
        if(qout.size()>(memory+1))
        {
           qout.pop_front();
        }
        for(int j=0;j<out.size();j++)
        {
           out[j].score = 0;
           out[j].lbl   = model.labels[j];
        }
        //----------------------------------------------------
        //add confidence levels
        //----------------------------------------------------
        for(int i=0;i<qout.size();i++)
        {
            for(int j=0;j<qout[i].size();j++)
            {
              out[j].score += qout[i][j];
            }
        }
        sort(out.begin(),out.end(),comp_func);
        //----------------------------------------------------
        //write the labels
        //---------------------------------------------------
        struct nv2_patch_label l;
        l.protocol_version = NV2_LABEL_PROTOCOL_VERSION;
        l.patch_id         = p.id;
        l.extra_info[0]    = 0;
		l.confidence       = (uint32_t)(NV2_MAX_LABEL_CONFIDENCE*tanh(out[NCLASS-1].score));
        snprintf(l.source, sizeof(l.source), "%s",id.c_str());
        //snprintf(l.name,sizeof(l.name),"CBCL");
        //snprintf(l.extra_info,sizeof(l.extra_info),"CBCL-2");
        snprintf(l.name,sizeof(l.name),out[NCLASS-1].lbl.c_str());
        snprintf(l.extra_info,sizeof(l.extra_info),out[NCLASS-2].lbl.c_str());
        if (l.patch_id % interval == 0)
        {
            nv2_label_server_send_label(s, &l);
            fprintf(stdout, "sent label '%s'\n", l.name);
        }
        else
        {
           fprintf(stdout, "DROPPED label '%s'\n", l.name);
        }
        nv2_image_patch_destroy(&p);
  }//end while
  nv2_label_server_destroy(s);
  /*cvReleaseVideoWriter(&pvdo);*/
  return 0;
}



import os
import json
import re
import cv2

def annotate(image,text_score,dt_box,filename):
  boxcount = 0
  path_name = "/content/gdrive/MyDrive/paddle_images_payslip/" \
               + filename + "_tnr" + ".txt"
  im = cv2.imread(image)
  big_array = []
  for text, score in text_score:
    with open(path_name,"a+") as f:
      f.write(text+"\n")
    dict_value = {"transcription":None, "points":None}
    if score> 0.97:
      text = text.rstrip()
      #print("{}, {:.3f}".format(text, score))
      dict_value["transcription"] = text
     
      x1 =  dt_box[boxcount][0][0]
      y1 = dt_box[boxcount][0][1]
      x2 = dt_box[boxcount][2][0]
      y2 = dt_box[boxcount][2][1]
      dict_value["points"] = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
      cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
      big_array.append(dict_value)
    boxcount+=1
  
  with open(os.path.join('/content/gdrive/MyDrive/paddle_images_payslip/train.txt'), 'a') as fh:
    fh.write("train_imgs/"+filename+".jpeg"+ "\t"+ json.dumps(str(big_array))+"\n")
      
  cv2.imwrite(os.path.join('/content/gdrive/MyDrive/paddle_images_payslip/train_imgs',filename + ".jpeg"),im)

def save_to_json(jsonoutfile,directory,jsondata):
  #if json file already exists:
  jsonout=directory+jsonoutfile+".json"
  #print(jsondata)
  if os.path.exists(jsonout):
    with open(jsonout) as f:
      data=json.load(f)
      data.update(jsondata)
    with open(jsonout, 'w') as f:
        json.dump(data, f) 
  else:
    with open(jsonout, "w") as outfile: 
      json.dump(jsondata,outfile) 
  
#function for net pay key and value
def findclosest_frnetpay(text,boxcount,rec_res,dt_boxes,mp_box):
  jsondata={}
  VALUE=""

  for i in range(0,len(mp_box)):
    if i!=boxcount:
      #finding boxes that are in the same row as netpay
      y_topleftcorner=dt_boxes[boxcount][3][1]
      y_bottonrightcorner=dt_boxes[boxcount][1][1]
      y_bottomleftcorner=dt_boxes[boxcount][0][1]
      y_toprightcorner=dt_boxes[boxcount][2][1]
      
      minimum=min(y_topleftcorner,y_bottonrightcorner,y_bottomleftcorner,y_toprightcorner)
      maximum=max(y_topleftcorner,y_bottonrightcorner,y_bottomleftcorner,y_toprightcorner)

      thresh=2.5
      #CHECK1-DOES Y COORD OF MIDPOINT LIE INBETWEEN Y COORD RANGE
      if mp_box[i][1]<=maximum+thresh and mp_box[i][1]>=minimum-thresh:
        #CHECK2-IS X COORD OF MIDPOINT GREATER THAN NETPAY BBOX XCOORD
        if mp_box[i][0]>mp_box[boxcount][0]:
          VALUE=rec_res[i][0]
          break
        else:
          #IF VALUE ISNT FOUND IN THE RIGHT --> LOOK DOWN
          for downbox in range(boxcount,len(mp_box)):
            if downbox!=boxcount:
              x_topleftcorner=dt_boxes[boxcount][3][0]
              x_bottonrightcorner=dt_boxes[boxcount][1][0]
              x_bottomleftcorner=dt_boxes[boxcount][0][0]
              x_toprightcorner=dt_boxes[boxcount][2][0]

              xmin=min(x_topleftcorner,x_bottonrightcorner,x_bottomleftcorner,x_toprightcorner)
              xmax=max((x_topleftcorner,x_bottonrightcorner,x_bottomleftcorner,x_toprightcorner))

              if mp_box[downbox][0]>=xmin and mp_box[downbox][0]<=xmax:
                VALUE=rec_res[downbox][0]
                break

  
  value_string = re.findall("\d+", rec_res[boxcount][0])
  if(len(value_string) and value_string[0].isnumeric()):
    VALUE = " ".join(value_string)
  jsondata["Net pay"]=VALUE
  return jsondata


#fn for getting midpoint of every bounding box
def bboxmidpoint(dt_boxes):
  mp_boxes=[]
  for bbox in dt_boxes:
    mpbbox=[]
    x_topleft=bbox[3][0]
    y_topleft=bbox[3][1]
    x_bot_r=bbox[1][0]
    y_bot_r=bbox[1][1]

    #midpoint of diagonal1
    x_mid1=abs(x_topleft-x_bot_r)/2
    y_mid1=abs(y_topleft-y_bot_r)/2

    x_topright=bbox[2][0]
    y_topright=bbox[2][1]
    x_bot_l=bbox[0][0]
    y_bot_l=bbox[0][1]

    #midpoint of diagonal2
    x_mid2=abs(x_topright-x_bot_l)/2
    y_mid2=abs(y_topright-y_bot_l)/2

    #averaging the two diagonal midpoints
    xdiff=(x_mid1+x_mid2)/2
    ydiff=(y_mid1+y_mid2)/2
    xmin=min(x_topleft,x_bot_r,x_topright,x_bot_l)
    ymin=min(y_topleft,y_bot_r,y_bot_l,y_topright)

    x_mid=xmin+xdiff
    y_mid=ymin+ydiff
    mpbbox.append(x_mid)
    mpbbox.append(y_mid)
    mp_boxes.append(mpbbox)

  return mp_boxes

#function to classify text as alpha,alphanumeric and numeric
def finding_texttype(text):
  #regex for num and alphapattern
  numpattern=re.compile("[\d,.\s]+$")
  alphpattern=re.compile("[a-zA-Z\s.,$()/-]+$")

  if alphpattern.match(text):
    text_type="ALPHA"

  elif numpattern.match(text):
    text_type="NUMERIC"
  else:
    text_type="ALPHANUMERIC"
  return text_type

#function to check if text is a date:
def date(text):
  months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July',
  'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  for month in months:
    if month in text:
      print('Date')
      break


#take text words as list
def get_organization_name(text,boxcount,rec_res,mp_boxes):
  mp_box_curr= mp_boxes[boxcount]
  orgisation_nm = {}
  company_string = []
  for i in range(0,len(mp_boxes)):
      if (abs(mp_box_curr[1]- mp_boxes[i][1])<1.2):
        company_string.append(rec_res[i][0])

  if company_string is None:
    return None
  else:
    orgisation_nm["Organization name"]= " ".join(company_string)
    return orgisation_nm

def extract_name(comp_emp_nam,comp_coord):
  if comp_emp_nam is None:
    return None
  else:
    val = False
    emp_name_coord = []
    print(comp_emp_nam)
    if(len(comp_emp_nam)>1):
      zip_nam = zip(comp_coord, comp_emp_nam)
      zip_nam = sorted(zip_nam)
      for j in range(0,len(list(zip_nam))):
        if(any(i.isdigit() for i in zip_nam[j][1])):
          val =True
          break   
      if val:
        zip_nam = list(zip_nam)[:j]
      comp_emp_nam = [i[1] for i in zip_nam][1:]
      comp_coord = [i[0] for i in zip_nam][1:]
      print(comp_emp_nam)
      print(comp_coord)
      emp_nm_track= [[comp_coord[0][0],comp_emp_nam[0]]]
      curr_coord=comp_coord[0][0]
      for i in range(1,len(comp_coord)):
        if (abs(comp_coord[i][0]-curr_coord) < 120):
          emp_nm_track.append([comp_coord[i][1],comp_emp_nam[i]])
      emp_nm_track = sorted(emp_nm_track,key= lambda x: x[1],reverse=True)
      emp_nm=[emp_nm_track[i][1] for i in range(0,len(emp_nm_track))]
      return " ".join(emp_nm)
    else:
      emp_name = [comp_emp_nam[0]]
      return emp_name[0].split(":")[1]


def get_emp_name_contains_name(text,boxcount,rec_res,mp_boxes):
  mp_box_curr= mp_boxes[boxcount]
  company_string_emp = [text]
  mid_point_comp = [mp_box_curr]
  emp_name ={}
  for i in range(0,len(mp_boxes)):
    if (abs(mp_box_curr[1]- mp_boxes[i][1])<25 and (mp_box_curr[0]<mp_boxes[i][0])):
      company_string_emp.append(rec_res[i][0])
      mid_point_comp.append(mp_boxes[i])
  name_emp = extract_name(company_string_emp,mid_point_comp)
  emp_name["Employee name"]= name_emp
  return emp_name

def get_emp_name_contains_mr(text,boxcount,rec_res,mp_boxes):
  mp_box_curr= mp_boxes[boxcount]
  company_string_emp = []
  mid_point_comp = []
  emp_name ={}
  '''this for loop will go through same y axis where 'name'
  word exist'''
  for i in range(0,len(mp_boxes)):
    if (abs(mp_box_curr[1]- mp_boxes[i][1])<1.2 and 
    ((mp_box_curr[0]<=mp_boxes[i][0]) and (mp_boxes[i][0]-mp_box_curr[0])<100)):
      company_string_emp.append(rec_res[i][0])
      mid_point_comp.append(mp_boxes[i])
  emp_name["Employee name"]= " ".join(company_string_emp)
  return emp_name

def get_emp_name_check(text,boxcount,rec_res,mp_boxes):
  mp_box_curr= mp_boxes[boxcount]
  company_string_emp = []
  mid_point_comp = [mp_box_curr]
  emp_name ={}
  for i in range(0,len(mp_boxes)):
    if (abs(mp_box_curr[1]- mp_boxes[i][1])<25 and
     (mp_box_curr[0]>mp_boxes[i][0]) and
     (mp_box_curr[0]-mp_boxes[i][0]<82) ):
      company_string_emp.append([mp_boxes[i],rec_res[i][0]])
  company_string_emp = sorted(company_string_emp,reverse=True)
  prev_name = ["employee"]
  if not len(company_string_emp):
    return True
  if company_string_emp[0][1].lower() in prev_name:
    return True
  else:
    return False


def get_employee_name(text,boxcount,rec_res,mp_boxes):
  if "name" in text.lower():
    #check if it is some other name like bank name or company name
    emp_name_check = get_emp_name_check(text,boxcount,rec_res,mp_boxes) 
    if emp_name_check:
      emp_name = get_emp_name_contains_name(text,boxcount,rec_res,mp_boxes)
    else:
      return None
  if "mr" in text.lower():
    emp_name = get_emp_name_contains_mr(text,boxcount,rec_res,mp_boxes)

  return emp_name

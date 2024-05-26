from classification import * 

patient_folder = 0
images = []
patient_folder, images = gfc(patient_folder, images)

for i in range(1,patient_folder+1):
    tlf = pd.read_csv('./testPatient/Patient_{}_Labels.csv'.format(i))
    tlf.Label[tlf.Label==2] = 1
    tlf.Label[tlf.Label==3] = 1
    tlf.to_csv('./testPatient/newPatient_{}_Labels.csv'.format(i), index=False)


lab_indices_0 = []
lab_indices_1 = []
for i in range(1,patient_folder+1):
    tlf = pd.read_csv('./testPatient/newPatient_{}_Labels.csv'.format(i))
    lab_count = images[i-1]
    l_index_0 = []
    l_index_1 = []
    for j in range (0, lab_count):
        cValue = tlf.iloc[j]['Label']
        if(cValue==0):
            l_index_0.append(j+1)
        else:
            l_index_1.append(j+1)
    lab_indices_0.append(l_index_0)
    lab_indices_1.append(l_index_1)


data_directory_path = os.path.join('./testPatient/','Data')
if(os.path.exists(data_directory_path)):
    shutil.rmtree(data_directory_path)
os.mkdir(data_directory_path)

data_value = patient_folder

for i in range(1, data_value+1):
    data_patient_path = os.path.join('./testPatient/Data','Patient_{}'.format(i))
    if(os.path.exists(data_patient_path)):
        shutil.rmtree(data_patient_path)
    os.mkdir(data_patient_path)

    source = './testPatient/Patient_{}'.format(i)
    destination =  './testPatient/Data/Patient_{}'.format(i)
    for pngFile in glob.iglob(os.path.join(source, '*thresh.png')):
        shutil.copy(pngFile, destination)
    
    label_path = './testPatient/newPatient_{}_Labels.csv'.format(i)
    shutil.copy(label_path, destination)


x_data = []
y_data = []
def createDataset():
    for i in range(1, data_value+1):
        img_count = images[i-1]
        for j in range(1, img_count+1):
            img_path = 'testPatient/Data/Patient_{}/IC_{}_thresh.png'.format(i, j)
            img_arr = cv.imread(img_path)
            img_arr=cv.resize(img_arr,(224,224))
            x_data.append(img_arr)

    data_x = np.array(x_data)/255.0

    for i in range(1, data_value+1):
        temp_label_values = pd.read_csv('./testPatient/Data/Patient_{}/newPatient_{}_Labels.csv'.format(i,i))
        t = temp_label_values.Label.values
        for i in range(0, len(t)):
            y_data.append(t[i])
    
    return data_x, y_data

data_x, y_data = createDataset()
data_pred = modlod(data_x, y_data)
create_results(images, data_pred)
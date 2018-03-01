#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "segmentation.h"
#include "optionparser.h"
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

Mat colorImg, src, grad, colorAnalysis, binImg, markers, markersCpy, imprev, seg;
Mat neighbors;
Point anchor;
double t;
int seSize = 11;
int mark_count = 1;
int dynamics = 0;
int connComptSize = 40;
int hDist = 5;
int vDist = 5;
int colorDist = 100;
bool betterMarkers = false;
Point prevPt(-1,-1);
Scalar currentcolor;
map<int, Scalar> colors;

enum optionType { WSHED=0, COLORSEG=1, VISCOUSFORCE=2 };
enum  optionIndex { UNKNOWN, HELP, INTERACTIVE, WSHEDLINE, ALGORITHM, SAVE };
const option::Descriptor usage[] =
{
	{UNKNOWN, 0, "", "",option::Arg::None, "Watershed Algorithms (testing)\nUSAGE: watershed <fname> [options]\n\n"
										"Options:" },
	{HELP, 0, "h", "help", option::Arg::None, "  --help, -h  \tPrint usage and exit." },
	{INTERACTIVE, 0, "i", "interactive", option::Arg::None, "  --interactive, -i  \tInteractive mode." },
	{WSHEDLINE, 0, "l", "line", option::Arg::None, "  --line, -l  \tShows watershed lines only." },
	{ALGORITHM, VISCOUSFORCE, "v", "viscous-force", option::Arg::None, "  --viscous-force, -v  \tViscous Force Watershed."},
	{ALGORITHM, COLORSEG, "r", "region-growing", option::Arg::None, "  --region-growing, -r  \tColor image segmentation (region growing) based on Watershed."},
	{ALGORITHM, WSHED, "w", "watershed", option::Arg::None, "  --watershed, -ws  \tClassical Watershed (default option)."},
	{SAVE, 0, "s", "save", option::Arg::None, "  --save, -s  \tSave segmentation results (.tif)." },
	{0,0,0,0,0,0}//sin esto la ayuda me da error ¿?
};

void segment(int algType, bool showLines=0)
{
	markersCpy = markers.clone();//utilizarlo en calculo de lineas de separacion
	int thick = 1 + (algType == VISCOUSFORCE);
	Mat srcTmp, markersTmp;
	copyMakeBorder(src, srcTmp, thick, thick, thick, thick, BORDER_REFLECT);
	copyMakeBorder(markers, markersTmp, thick, thick, thick, thick, BORDER_REFLECT);

	t = static_cast<double>(getTickCount());
	switch (algType)//estos son los mas importantes
	{
	case VISCOUSFORCE:
		imgseg::vfWatershed(srcTmp, markersTmp);
		break;
	case COLORSEG:
		imgseg::watershedColor(srcTmp, markersTmp);
		break;
	case WSHED:		
		imgseg::watershed(srcTmp, markersTmp);
		break;
	default:
		break;
	}
	t = static_cast<double>(getTickCount()) - t;
	cout << "execution time: " << t*1000./getTickFrequency() << "ms\n";

	//para eliminar rectangulo exterior que se crea en los algoritmos
	auto it = markers.begin<int>();
	for (int i = thick; i < markersTmp.rows - thick; i++)
	{
		int *tmpLbl = markersTmp.ptr<int>(i) + thick;
		for (int j = thick; j < markersTmp.cols - thick; j++, tmpLbl++, it++)
			*it = *tmpLbl;
	}

	//preparando resultados finales
	if(showLines)
		if(algType == WSHED)
			seg = markers == -1;//cambiar a '!= -1' para fondo blanco y lineas negras
		else
		{
			markers.convertTo(seg, CV_8U);
			imgseg::watershed(seg, markersCpy);
			seg = markersCpy == -1;//cambiar a '!= -1' para fondo blanco y lineas negras
		}
	else
	{
		seg = Mat(markers.size(), CV_8UC3);
		imgseg::fillRegions(markers, seg);	
	}
}

static void onDynamicsTrackbar(int, void*)
{	
	betterMarkers = false;
	Mat reconst = grad + dynamics;
	imgseg::dualReconstruction(grad, reconst);
	compare(reconst, grad, binImg, CMP_GT);
	imshow("viewer", binImg);		
}

void showMarkers()
{
	//visualizando los marcadores
	Mat imposedMarkers;
	colorImg.copyTo(imposedMarkers);

	auto marker_it = markers.begin<int>();
	auto marker_itend = markers.end<int>();
	auto img_it = imposedMarkers.begin<Vec3b>();

	map<int, Vec3b> markerColor;
	for (; marker_it != marker_itend; marker_it++, img_it++)
	{
		if (*marker_it)
		{
			if(!markerColor.count(*marker_it))
				markerColor[*marker_it] = Vec3b(rand()&255, rand()&255, rand()&255);
			*img_it = markerColor.at(*marker_it);
		}
	}

	imshow("viewer", imposedMarkers);
}

static void onMouse(int ev, int x, int y, int flags, void*)
{
    int thick = 2;

    if(x < thick || x >= src.cols - thick || y < thick || y >= src.rows - thick)
        return;

    if(ev == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON))
        prevPt = Point(-1,-1);
    else if(ev == CV_EVENT_LBUTTONDOWN)
    {
		printf("current marker: %d\n", mark_count++);
        prevPt = Point(x,y);
		if(colors.find(mark_count) == colors.end())
		{
			currentcolor = Scalar(rand()&255, rand()&255, rand()&255);
			colors[mark_count] = currentcolor;
		}
		else
			currentcolor = colors.at(mark_count);
    }
    else if(ev == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
    {
        Point pt(x, y);
        if(prevPt.x < 0)
            prevPt = pt;
        line(markers, prevPt, pt, Scalar(mark_count), thick, 8, 0);
        line(imprev, prevPt, pt, currentcolor, thick, 8, 0);
        prevPt = pt;
        imshow("viewer", imprev);
    }
}

string getFileName(string fname)
{
	string name = fname;
	const size_t last_slash_idx = name.find_last_of("\\/");
	if (std::string::npos != last_slash_idx)
		name.erase(0, last_slash_idx + 1);

	const size_t period_idx = name.rfind('.');
	if (std::string::npos != period_idx)
		name.erase(period_idx);

	return name;
}

int main(int argc, char** argv)
{
	
	argc-=(argc>0); argv+=(argc>0);
	option::Stats  stats(usage, argc, argv);
	option::Option* options = new option::Option[stats.options_max];
	option::Option* buffer  = new option::Option[stats.buffer_max];
	option::Parser parse(usage, argc, argv, options, buffer);
	
	if (parse.error())
	{
		cout << "parsing error." << endl;
		option::printUsage(std::cout, usage);
		return 1;
	}
	
	if (options[HELP]) 
	{
		option::printUsage(std::cout, usage);
		return 0;
	}

	int algType = options[ALGORITHM].last()->type();

	string filename;// = "..\\.\\images\\objetos.jpg";
	cout << "enter image file name: ";
	getline(cin, filename);
	cout << endl;

	colorImg = imread(filename);
	if(colorImg.empty())
    {
        cout << "error: no image loaded." << endl;
        return 0;
    }

	if (algType == COLORSEG)
		src = colorImg.clone();
	else
		cvtColor(colorImg, src, CV_BGR2GRAY);

	markers = Mat::zeros(src.size(), CV_32S);
	
	if (options[INTERACTIVE])
	{
		if(algType != COLORSEG)
		{
			imprev = Mat(src.size(), CV_8UC3);//para visualizar marcadores con diferentes colores
			cvtColor(src, imprev, CV_GRAY2BGR);
		}
		else
			imprev = src.clone();

		namedWindow("viewer");
		setMouseCallback("viewer", onMouse, 0);		
		imshow("viewer", imprev);

		cout << "press 's' key to segment marked objects." << endl;
		cout << "use '+' or '-' for marker selection." << endl;

		while(true)
		{
			char w = waitKey();
        
			if((char)w == 's')
				break;

			if((char)w == '+')
				mark_count++;

			if((char)w == '-')
			{
				mark_count--;
				if(!mark_count)
					mark_count++;//siempre positivo
			}

			printf("current marker set to: %d\n", mark_count);
		}

		if(options[WSHEDLINE])//mostrar lineas de separacion
		{			
			segment(algType, true);
			imshow("viewer", seg);
			cout << "segmented image..." << endl;			
		}
		else
		{
			segment(algType, false);
			Mat segView(markers.size(), CV_8UC3);
			addWeighted(colorImg, 0.2, seg, 0.8, 0, segView);
			imshow("viewer", segView);
			cout << "segmented image..." << endl;			
		}
		waitKey();
	}
	else
	{
		cvtColor(colorImg, colorAnalysis, CV_BGR2Lab);//para obtener mejor metrica de similitud de los colores
		if(algType == COLORSEG)
			cvtColor(src, imprev, CV_BGR2GRAY);//la reconstruccion es en escala de grises, se esta usando imprev de comodin
		else
			imprev = src.clone();

		Mat se = getStructuringElement(MORPH_CROSS, Size(3, 3));
		morphologyEx(imprev, grad, MORPH_GRADIENT, se);
		
		namedWindow("viewer");
		namedWindow("trackbars");
		createTrackbar("dynamics",  "trackbars", &dynamics, 255, onDynamicsTrackbar);
		createTrackbar("connSize",  "trackbars", &connComptSize, 200);
		createTrackbar("colorDist", "trackbars", &colorDist, 800);
		createTrackbar("hDist",     "trackbars", &hDist, 20);
		createTrackbar("vDist",     "trackbars", &vDist, 20);
		createTrackbar("seSize",    "trackbars", &seSize, 21);
				
		Mat imgTrackbar;
		int scaledRows = colorImg.rows;
		int scaledCols = colorImg.cols;
		if (scaledRows >= 321 || scaledCols >= 481)//con el size de las imgs de la Berkeley se salen de mi display
		{
			scaledRows = static_cast<int>(scaledRows / 1.5);
			scaledCols = static_cast<int>(scaledCols / 1.5);
		}
		
		resize(colorImg, imgTrackbar, Size(scaledCols, scaledRows));
		imshow("trackbars", imgTrackbar);
		imshow("viewer", src);

		cout << "press 's' key to segment or another key to exit." << endl;
		cout << "press 'm' to get color markers." << endl;

		while (true)
		{
			char w = waitKey();

			if((char)w == 'm')
			{
				betterMarkers = true;
				markers = Scalar::all(0);
				Mat labels(src.size(), CV_32S);
				map<int, vector<Point>> connCompts = imgseg::getConnectedComponents(binImg, labels);	
				seSize = (seSize <= 1)? 3 : ((seSize % 2 == 0)? seSize + 1 : seSize);
				neighbors = getStructuringElement(MORPH_ELLIPSE, Size(seSize, seSize));//actualizar anchor abajo
				anchor = Point(seSize / 2, seSize / 2);

				imgseg::getMarkers(colorAnalysis, binImg, markers, neighbors, anchor, connComptSize, colorDist, hDist, vDist);
				showMarkers();
			}
			else if((char)w == 's')
			{
				//Utilizar solo con las componentes conexas de binImg como marcadores
				if(!betterMarkers)
				{
					markers = Scalar::all(0);
					map<int, vector<Point>> connCompts = imgseg::getConnectedComponents(binImg, markers);
				
					//eliminando componentes conexas menor que un size
					imgseg::removeConnCompts(connCompts, connComptSize);
					//refinando markers
					markers = Scalar::all(0);
					auto itmap = connCompts.begin();
					auto itendmap = connCompts.end();
					for(; itmap != itendmap; itmap++)
					{
						auto itpoint = (itmap->second).begin();
						auto itpointend = (itmap->second).end();

						for(; itpoint != itpointend; itpoint++)
							*(markers.ptr<int>(itpoint->x) + itpoint->y) = itmap->first;
					}					
				}
				
				if(options[WSHEDLINE])//mostrar lineas de separacion
				{
					segment(algType, true);
					imshow("viewer", seg);
					cout << "segmented image..." << endl;
				}
				else
				{
					segment(algType, false);
					Mat segView(markers.size(), CV_8UC3);
					addWeighted(colorImg, 0.2, seg, 0.8, 0, segView);
					imshow("viewer", segView);
					cout << "segmented image..." << endl;
				}
			}
			else
				break;
		}
	}
	
	if(options[SAVE] && !seg.empty())
		imwrite(getFileName(filename) + "_seg.tif", seg);

	delete[] options;
	delete[] buffer;
	destroyAllWindows();//*/

	//////////////////////////////////////WATERSHED//////////////////////////////////////
	/*
    //Cargando imagen
    string filename = "..\\.\\images\\ferrari.jpg";
    
    if(argc >= 2)
        filename = string(argv[1]);

    src = imread(filename, 0);//escala de grises	
    //src = imread(filename);//en colores para probar imgseg::watershedColor

	if(src.empty())
    {
        cout << "no image loaded." << endl;
        return 0;
    }	
	
	if(src.type() == CV_8U)//en escala de grises
		cvtColor(src, imprev, CV_GRAY2BGR);
	else//en colores CV_8UC3
		imprev = src.clone();

    namedWindow("viewer");
    setMouseCallback("viewer", onMouse, 0);
    imshow("viewer", imprev);

	//preparando marcadores
	markers = Mat::zeros(src.size(), CV_32S);

    cout << "'s' -> watershed" << endl;

    while(true)
    {
        char w = waitKey();
        
        if((char)w == 's')
            break;

        if((char)w == '+')
            mark_count++;

        if((char)w == '-')
            mark_count--;

		printf("current marker set to: %d\n", mark_count);
    }	
	
	//gradiente de la imagen
	//morphologyEx(src, grad, MORPH_GRADIENT, getStructuringElement(MORPH_CROSS, Size(3,3)));//trabajando con el gradiente de la imagen
	//imshow("viewer", grad);
	//waitKey();
	
	//modificacion homotopica del gradiente
	//Mat hmod;
	//imgseg::homotopyModification(grad, markers, hmod);
	//imshow("viewer", hmod);
	//waitKey();

	///////PARA EVALUACION (comentar el resto del codigo)///////
	//imwrite(getFileName(filename) + "_markers.tif", imprev);
	//src = imread(filename);//en colores
	//segment(1);//wshed Color
	//imwrite(getFileName(filename) + "_wshedColor_seg.tif", seg);
	//markers = markersCpy.clone();
	//src = imread(filename, 0);//en grises
	//segment(2);//wshed Viscosidad
	//imwrite(getFileName(filename) + "_wshedVF_seg.tif", seg);
	//imshow("viewer", seg);
	//waitKey();
	//////////////////////////////////////////////////////////////
	
    t = static_cast<double>(getTickCount());
	
	//imgseg::gradWshed(grad, markers);
	imgseg::watershed(src, markers);
	//imgseg::watershedColor(src, markers);//cargar la imagen en colores	
	//imgseg::topographicDistWshed(hmod, markers);
	//imgseg::vfWatershed(src, markers);	

    t = static_cast<double>(getTickCount()) - t;
    cout << "execution time = " << t*1000./getTickFrequency() << "ms\n";

    //mostrando resultados
	//Mat wshed = markers != -1;//solo para imgseg::watershed o imgseg::gradWshed que son los que computan las lineas de separacion
	//imshow("viewer", wshed);
	//waitKey();
	
	Mat reg(markers.size(), CV_8UC3);
	imgseg::fillRegions(markers, reg);//, Vec3b(0,255,0) como 3er parametro para dibujar lineas de separacion en verde, por ejemplo
	imshow("viewer", reg);
    waitKey();//*/

    return 0;
}

#pragma once
#include <vector>
#include "segmentation.h"
#include "hierarchical_queue.h"

#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>

using namespace cv;

namespace imgseg
{
	/*Implementaciones de la transformada de Watershed*/

    void gradWshed(Mat &grad, Mat &markers)
    {
        //aqui falta verificar que src sea en escala de grises (CV_8U)
        //que markers sea de tipo de 32bit-signed, 1-channel (CV_32SC1)
        //y que las imagenes tengan el mismo size
        
        const int WSHED  = -1;
        const int QUEUED = -2;
		
        const int rows = grad.rows;
        const int cols = grad.cols;

        //variables
		Point p;
        int r, c, label, labelAux;
        int *px, *pxAux;

        //cola jerarquica   
        HQueue<Point, 256> hq(rows*cols);
        
        //lambdas
        auto pushPixel = [&](int x, int y)
        {
            if(!*px)
            {
                hq.Push(Point(x,y), *(grad.ptr<uchar>(x) + y));
                *px = QUEUED;
            }
        };

        auto pushAll = [&]()
        {
            px = markers.ptr<int>(r - 1) + c;//up
            pushPixel(r - 1, c);            

            px = pxAux + 1;//right
            pushPixel(r, c + 1);

            px = markers.ptr<int>(r + 1) + c;//down
            pushPixel(r + 1, c);

            px = pxAux - 1;//left
            pushPixel(r, c - 1);
        };
    
        auto labelPixel = [&]()
        {            
            labelAux = *px;

            if(labelAux > 0)
                if(!label)
                    label = labelAux;
                else if(label != labelAux)
                    label = WSHED;
        };        

		rectangle(markers, Point(0,0), Point(cols-1, rows-1), Scalar(WSHED));//truco

        /********insertando en la cola los pixeles vecinos de cada marcador********/
        for(r = 1; r < rows; r++)
        {
            pxAux = markers.ptr<int>(r) + 1;
            for(c = 1; c < cols; c++, pxAux++)
                if(*pxAux > 0)
                    pushAll();
        }        
		
        /********************************inundacion********************************/
        while(!hq.Empty())
        {
            p = hq.Pop();
            r = p.x;
            c = p.y;

            label = 0, labelAux = 0;
            pxAux = markers.ptr<int>(r) + c;
            
            px = markers.ptr<int>(r - 1) + c;//up
            labelAux = *px;
            if(labelAux > 0) label = labelAux;

            px = pxAux + 1;//right
            labelPixel();

            px = markers.ptr<int>(r + 1) + c;//down
            labelPixel();

            px = pxAux - 1;//left
            labelPixel();

            *pxAux = label;
            if(label == WSHED)
                continue;
                                    
            pushAll();
        }
    }

	void watershed(Mat &img, Mat &markers)
    {
        //aqui falta verificar que src sea en escala de grises (CV_8U)
        //que markers sea de tipo de 32bit-signed, 1-channel (CV_32SC1)
        //y que las imagenes tengan el mismo size
        
        const int WSHED  = -1;
        const int QUEUED = -2;

        const int rows = img.rows;
        const int cols = img.cols;

        //variables
		Point p;
        int r, c, label, labelAux;
        int *px, *pxAux;

        //cola jerarquica
        HQueue<Point, 256> hq(rows*cols);

		#define updateIntensity(intensity)\
		{								\
			if (intensity > neighbMax)	\
				neighbMax = intensity;	\
			if (intensity <  neighbMin)	\
				neighbMin = intensity;	\
		}
        
        //lambdas
        auto pushPixel = [&](int x, int y)
        {
            if(!*px)
            {		
				uchar *srcPtr = img.ptr<uchar>(x) + y;
				uchar neighbMax, neighbMin;
				neighbMax = neighbMin = *(srcPtr + 1);

				updateIntensity(*(srcPtr - 1));
				updateIntensity(*(srcPtr + img.step));
				updateIntensity(*(srcPtr - img.step));

				hq.Push(Point(x,y), neighbMax - neighbMin);
                *px = QUEUED;
            }
        };

        auto pushAll = [&]()
        {
            px = markers.ptr<int>(r - 1) + c;//up
            pushPixel(r - 1, c);            

            px = pxAux + 1;//right
            pushPixel(r, c + 1);

            px = markers.ptr<int>(r + 1) + c;//down
            pushPixel(r + 1, c);

            px = pxAux - 1;//left
            pushPixel(r, c - 1);
        };
    
        auto labelPixel = [&]()
        {            
            labelAux = *px;

            if(labelAux > 0)
                if(!label)
                    label = labelAux;
                else if(label != labelAux)
                    label = WSHED;
        };        

		rectangle(markers, Point(0,0), Point(cols-1, rows-1), Scalar(WSHED));//truco

        /********insertando en la cola los pixeles vecinos de cada marcador********/
        for(r = 1; r < rows; r++)
        {
            pxAux = markers.ptr<int>(r) + 1;
            for(c = 1; c < cols; c++, pxAux++)
                if(*pxAux > 0)
                    pushAll();
        }  

		//hq.SetLock(true);
        
        /********************************inundacion********************************/
        while(!hq.Empty())
        {
            p = hq.Pop();
            r = p.x;
            c = p.y;

            label = 0, labelAux = 0;
            pxAux = markers.ptr<int>(r) + c;
            
            px = markers.ptr<int>(r - 1) + c;//up
            labelAux = *px;
            if(labelAux > 0) label = labelAux;

            px = pxAux + 1;//right
            labelPixel();

            px = markers.ptr<int>(r + 1) + c;//down
            labelPixel();

            px = pxAux - 1;//left
            labelPixel();

            *pxAux = label;
            if(label == WSHED)
                continue;
                                    
            pushAll();
        }
    }		

	void watershedColor(Mat &img, Mat &markers)
    {
        //aqui falta verificar que src sea en escala de grises (CV_8U)
        //que markers sea de tipo de 32bit-signed, 1-channel (CV_32SC1)
        //y que las imagenes tengan el mismo size
        
		using std::max;
		using std::min;

        const int rows = img.rows;
        const int cols = img.cols;

		//variables
		int *px, *label;
		int r, c;
		Vec3b color;//colores (b,g,r)
		//int intensity;//bit-mixing

        //cola jerarquica
        HQueue<Point, 256> hq(rows*cols);
        //MapHQueue<Point, int> hq(rows*cols);//bit-mixing

		//probando mit-mixing
		//Mat tmp = img.clone();
		//img = Mat(tmp.size(), CV_32S);
		//bitMixingImg(tmp, img);//estoy cambiando el tipo de dato de img a int
        
        //lambdas
		auto ucharAbs = [](uchar a, uchar b)->uchar
		{
			return (a > b)? a - b : b - a;
		};

        auto pushPixel = [&](int x, int y)
        {
			//prioridad por los colores
			Vec3b nextColor = *(img.ptr<Vec3b>(x) + y);
			uchar bdiff = ucharAbs(color[0], nextColor[0]);
			uchar gdiff = ucharAbs(color[1], nextColor[1]);
			uchar rdiff = ucharAbs(color[2], nextColor[2]);

			uchar diff = max<uchar>(bdiff, gdiff);
			diff = max<uchar>(diff, rdiff);
			hq.Push(Point(x, y), diff);			

			//prioridad con variante bit-mixing, utilizar MapHQueue
			//hq.Push(Point(x,y), std::abs(intensity - *(img.ptr<int>(x) + y)));
        };

		auto labelPixel = [&](int x, int y)
		{
			if(!*px)
			{
				*px = *label;
				pushPixel(x, y);
			}
		};

		rectangle(markers, Point(0,0), Point(cols-1, rows-1), Scalar(-1));//truco

        /********insertando en la cola los pixeles vecinos de cada marcador********/
        for(r = 0; r < rows; r++)
        {
            label = markers.ptr<int>(r);
			for(c = 0; c < cols; c++, label++)
                if(*label > 0)
				{
					color = *(img.ptr<Vec3b>(r) + c);//descomentar para colores
					//intensity = *(img.ptr<int>(r) + c);//bit-mixing
					pushPixel(r, c);
				}
        }        
        
        /********************************inundacion********************************/
        while(!hq.Empty())
        {
            Point p = hq.Pop();
            r = p.x;
            c = p.y;

            label = markers.ptr<int>(r) + c;   
			color = *(img.ptr<Vec3b>(r) + c);//descomentar para colores
			//intensity = *(img.ptr<int>(r) + c);//bit-mixing

            px = markers.ptr<int>(r - 1) + c;//up
			labelPixel(r - 1, c);

            px = label + 1;//right
			labelPixel(r, c + 1);

            px = markers.ptr<int>(r + 1) + c;//down
			labelPixel(r + 1, c);

            px = label - 1;//left
			labelPixel(r, c - 1);
        }
    }		

	void vfWatershed(Mat &img, Mat &markers)
	{
		//aqui falta verificar que src sea en escala de grises (CV_8U)
		//que markers sea de tipo de 32bit-signed, 1-channel (CV_32SC1)
		//y que las dos matrices tengan el mismo size        

		const float c = 0.005f;//constante de viscosidad
		const float cInv = 1 / c;
    
		int rows = img.rows;
		int cols = img.cols;		
		//cola jerarquica     		
		MapHQueue<Point, float> hq(rows * cols);

		//matrices utiles
		Mat origMarkers = markers.clone();
		rectangle(origMarkers, Point(0,0), Point(--cols, --rows), Scalar(1), 2);//para no salirse de rango
    
		Mat topographicDist;
		morphologyEx(img, topographicDist, MORPH_GRADIENT, getStructuringElement(MORPH_RECT, Size(3,3)));

		/********insertando en la cola los pixeles vecinos de cada marcador********/
    
		--rows; --cols;//aqui es donde ultimo se utilizan, esto de acuerdo a lo del rectangulo en origMarkers
		for(int x = 2; x < rows; x++)
		{
			int *label = markers.ptr<int>(x) + 2;
			uchar *dist = topographicDist.ptr<uchar>(x) + 2;
			for(int y = 2; y < cols; y++, label++, dist++)
				if(*label && !(*(label - 1) && *(label + 1) && *(markers.ptr<int>(x - 1) + y) && *(markers.ptr<int>(x + 1) + y)))
					hq.Push(Point(x, y), *dist);
		}

		//variables		
		int   *label;
		int   *origLabel;
		uchar *neighbDist;
		float vf;
		int   currentLabel;

		#define updateDist(dx, dy)							\
		{													\
			if(!*origLabel && !*label)						\
			{												\
				*label = currentLabel;						\
				hq.Push(Point(dx, dy), *neighbDist + vf);	\
			}												\
		}

		/********************************inundacion********************************/
		while(!hq.Empty())
		{
			Point p = hq.Pop();
			int x = p.x;
			int y = p.y;

			//calculando viscosidad
			/*float *neighbDist = topographicDist.ptr<float>(x - 1) + (y - 1);
			float sum = (1 - 1 / std::exp(c * *(neighbDist++)));
			sum += (1 - 1 / std::exp(c * *(neighbDist++)));
			sum += (1 - 1 / std::exp(c * *neighbDist));
        
			neighbDist = topographicDist.ptr<float>(x) + (y - 1);
			sum += (1 - 1 / std::exp(c * *neighbDist));
			neighbDist += 2;
			sum += (1 - 1 / std::exp(c * *neighbDist));

			neighbDist = topographicDist.ptr<float>(x + 1) + (y - 1);
			sum += (1 - 1 / std::exp(c * *(neighbDist++)));
			sum += (1 - 1 / std::exp(c * *(neighbDist++)));
			sum += (1 - 1 / std::exp(c * *neighbDist));*/

			neighbDist = topographicDist.ptr<uchar>(x - 1) + (y - 1);
			float sum = 8;
			sum -=	1 / std::exp(c * *(neighbDist++));
			sum -=  1 / std::exp(c * *(neighbDist++));
			sum -=  1 / std::exp(c * *neighbDist);
        
			neighbDist += topographicDist.step - 2;
			sum -= 1 / std::exp(c * *neighbDist);
			neighbDist += 2;
			sum -= 1 / std::exp(c * *neighbDist);

			neighbDist += topographicDist.step - 2;
			sum -= 1 / std::exp(c * *(neighbDist++));
			sum -= 1 / std::exp(c * *(neighbDist++));
			sum -= 1 / std::exp(c * *neighbDist);

			vf = (cInv * std::log(1 + sum)) / 2;
			currentLabel = *(markers.ptr<int>(x) + y);
        
			//fila superior
			int dx = x - 1;
			int dy = y - 1;
			neighbDist = topographicDist.ptr<uchar>(dx) + dy;
			origLabel = origMarkers.ptr<int>(dx) + dy;
			label = markers.ptr<int>(dx) + dy;
			updateDist(dx, dy);

			dy++;
			neighbDist++;
			origLabel++;
			label++;
			updateDist(dx, dy);

			dy++;
			neighbDist++;
			origLabel++;
			label++;
			updateDist(dx, dy);

			//fila actual
			dx++;
			dy -= 2;
			neighbDist += topographicDist.step - 2;
			origLabel = origMarkers.ptr<int>(dx) + dy;
			label = markers.ptr<int>(dx) + dy;
			updateDist(dx, dy);

			dy += 2;
			neighbDist += 2;
			origLabel += 2;
			label += 2;
			updateDist(dx, dy);

			//fila inferior
			dx++;
			dy -= 2;
			neighbDist += topographicDist.step - 2;
			origLabel = origMarkers.ptr<int>(dx) + dy;
			label = markers.ptr<int>(dx) + dy;
			updateDist(dx, dy);
        
			dy++;
			neighbDist++;
			origLabel++;
			label++;
			updateDist(dx, dy);

			dy++;
			neighbDist++;
			origLabel++;
			label++;
			updateDist(dx, dy);
		}
	}	

	void topographicDistWshed(Mat &grad, Mat &markers)
    {
        //aqui falta verificar que src sea CV_32F
        //que markers sea de tipo de 32bit-signed, 1-channel (CV_32SC1)
		//que neighb no contenga a anchor
        //y que las imagenes tengan el mismo size

		//chamfer neighborhood
		//vector<Point> neighb;
		//Mat chamfer(5, 5, CV_8U, Scalar::all(1));
		//chamfer.at<uchar>(2, 2) = 0;//centro		
		//chamfer.at<uchar>(0, 0) = 0;
		//chamfer.at<uchar>(0, 2) = 0;
		//chamfer.at<uchar>(0, 4) = 0;
		//chamfer.at<uchar>(2, 0) = 0;
		//chamfer.at<uchar>(2, 4) = 0;
		//chamfer.at<uchar>(4, 0) = 0;
		//chamfer.at<uchar>(4, 2) = 0;
		//chamfer.at<uchar>(4, 4) = 0;
		//const int anchorX = 2;
		//const int anchorY = 2;
		//for (int x = 0; x < chamfer.rows; x++)
		//	for (int y = 0; y < chamfer.cols; y++)
		//		if (*(chamfer.ptr<uchar>(x) + y))
		//			neighb.push_back(Point(x,y));

		//vecindad clasica
		const Point points[] = { Point(0,0), Point(0,1), Point(0,2), Point(1,2), Point(2,2), Point(2,1), Point(2,0), Point(1,0) };//8 vecinos
		//const Point points[] = { Point(1,1), Point(0,0), Point(1,2), Point(0,1), Point(0,2), Point(0,3), Point(1,3), Point(0,4), 
		//				         Point(2,3), Point(1,4), Point(2,4), Point(3,4), Point(3,3), Point(4,4), Point(3,2), Point(4,3), 
		//						 Point(4,2), Point(4,1), Point(3,1), Point(4,0), Point(2,1), Point(3,0), Point(2,0), Point(1,0) };//24 vecinos
		const vector<Point> neighb(points, points + 8);//+ 24
		const int anchorX = 1;
		const int anchorY = 1;

        const int rows = grad.rows;
        const int cols = grad.cols;

        //matrices utiles
		Mat dist(grad.size(), CV_32F, Scalar::all(FLT_MAX));
		Mat ls(grad.size(), CV_32F, Scalar::all(-1));
		Mat vf(grad.size(), CV_32F, Scalar::all(-1));

        //cola jerarquica
		//OrderedQueue<Point, float> hq;
		MapHQueue<Point, float> hq(rows * cols);

		 //lambdas
		auto validPos = [&](int x, int y)
		{
			return (x >= 0 && x < rows && y >= 0 && y < cols);
		};

		auto lowerSlope = [&](int x0, int y0)->float
		{
			float *curLowerSlope = ls.ptr<float>(x0) + y0;
			
			if (*curLowerSlope < 0)//no ha sido calculado
			{			
				float intensity = *(grad.ptr<uchar>(x0) + y0);
				float slope = 0;

				auto it = neighb.begin();
				auto itend = neighb.end();
				for (int x1, y1; it != itend; it++)
				{
					x1 = x0 + (it->x - anchorX);
					y1 = y0 + (it->y - anchorY);
					if (validPos(x1, y1) && intensity >  *(grad.ptr<uchar>(x1) + y1))
						slope = std::max(slope, (intensity - *(grad.ptr<uchar>(x1) + y1)) / (std::abs(x1 - x0) + std::abs(y1 - y0)));
				}
				*curLowerSlope = slope;
			}

			return *curLowerSlope;
		};

		auto topographicDist = [&](int x0, int y0, int x1, int y1)->float
		{
			float intensity0 = *(grad.ptr<uchar>(x0) +  y0);
			float intensity1 = *(grad.ptr<uchar>(x1) +  y1);

			if (intensity0 == intensity1)
				return 0.5f*(lowerSlope(x0, y0) + lowerSlope(x1, y1))*(std::abs(x0 - x1) + std::abs(y0 - y1));

			return (intensity0 > intensity1)? lowerSlope(x0, y0) : lowerSlope(x1, y1);
		};

		auto viscousForce = [&](int x0, int y0)->float
		{
			float *curViscousForce = vf.ptr<float>(x0) + y0;
			if (*curViscousForce < 0)//no ha sido calculado
			{
				const float c = 0.005f;
				//float logSum = neighbCount;
				float logSum = 0;

				auto it = neighb.begin();
				auto itend = neighb.end();
				for (int x1, y1; it != itend; it++)
				{
					x1 = x0 + (it->x - anchorX);
					y1 = y0 + (it->y - anchorY);
					if (validPos(x1, y1))
						//logSum -= std::exp(-c * topographicDist(x0, y0, x1, y1));
						logSum += (1 - 1 / std::exp(c * topographicDist(x0, y0, x1, y1)));
				}
				*curViscousForce = (1 / c) * std::log(1 + logSum);
			}
			
			return *curViscousForce;
		};
				
        /********inicializacion********/
        for(int x = 1; x < rows; x++)
        {
            int *label = markers.ptr<int>(x) + 1;
            for(int y = 1; y < cols; y++, label++)
				if(*label)
				{
					float intensity = *(grad.ptr<uchar>(x) + y);
					*(dist.ptr<float>(x) + y) = intensity;

					//los píxeles del borde del mínimo se introducen en la cola con distancia topografica f(x)
					if (!*(label - 1) || !*(label + 1) || !*(markers.ptr<int>(x - 1) + y) || !*(markers.ptr<int>(x + 1) + y))
						hq.Push(Point(x, y), intensity);
				}
        }

        /********************************inundacion********************************/		
        while(!hq.Empty())
        {
			Point p = hq.Pop();
			int x0 = p.x;
            int y0 = p.y;

			float pDist = *(dist.ptr<float>(x0) + y0);

			auto it = neighb.begin();
			auto itend = neighb.end();
			for (int x1, y1; it != itend; it++)
			{
				x1 = x0 + (it->x - anchorX);
				y1 = y0 + (it->y - anchorY);

				if(validPos(x1, y1))
				{
					int *label = markers.ptr<int>(x1) + y1;

					float *neighbDist = dist.ptr<float>(x1) + y1;
					float cost = topographicDist(x0, y0, x1, y1);//*/ + viscousForce(x0, y0)/2;

					bool labeled = *label > 0;
					if(*neighbDist > pDist + cost)
					{
						*neighbDist = pDist + cost;
						*label = *(markers.ptr<int>(x0) + y0);
						if(!labeled)
							hq.Push(Point(x1, y1), *neighbDist);
					}
				}				
			}
        }
    }	
	
	/*Operadores y Utiles*/

    void dualReconstruction(Mat &mask, Mat &marker)
    {
        //verificar que src.size() == marker.size() y sean CV_8U

        const uchar FINAL  = 1;
        const uchar QUEUED = 2;

        int rows = mask.rows;
        int cols = mask.cols;

		//variables
		Point p;
		int r, c;
		uchar *markerPtr;
		uchar *maskPtr;
		uchar *statPtr;
        uchar h;
        Mat status = Mat::zeros(mask.size(), CV_8U);//todos como CANDIDATE = 0 inicialmente
        HQueue<Point, 256> hq(2*(rows*cols));//doble del tamanno de la imagen para garantizar memoria                

        rectangle(status, Point(0, 0), Point(--cols, --rows), Scalar(FINAL));//truco		

        //inicializacion        
        for(r = 1; r < rows; r++)
        {
            markerPtr = marker.ptr<uchar>(r) + 1;
			maskPtr = mask.ptr<uchar>(r) + 1;
			for(c = 1; c < cols; c++, markerPtr++, maskPtr++)
            {
                h = *markerPtr;
				*markerPtr = std::max(h, *maskPtr);
                hq.Push(Point(r, c), *markerPtr);//h o marker.at<uchar>(i, j)?
            }
        }

        auto push_neighbor = [&](int x, int y)
        {
            statPtr = status.ptr<uchar>(x) + y;
            if(!(*statPtr))
            {
				h = std::max(*markerPtr, *(mask.ptr<uchar>(x) + y));
                *(marker.ptr<uchar>(x) + y) = h;
                hq.Push(Point(x, y), h);
                *statPtr = QUEUED;
            }
        };

        //reconstruccion por erosion
        while(!hq.Empty())
        {
            p = hq.Pop();
			r = p.x;
			c = p.y;

            statPtr = status.ptr<uchar>(r)+ c;

            if(*statPtr == FINAL)
                continue;

            *statPtr = FINAL;

			markerPtr = marker.ptr<uchar>(r) + c;
            push_neighbor(r - 1, c);
            push_neighbor(r, c + 1);
            push_neighbor(r + 1, c);
            push_neighbor(r, c - 1);
        }
    }

    void getMinima(Mat &img, Mat &minima, int h)
    {
		if (!minima.data)
			minima = Mat(img.size(), CV_8U);

        minima = img + h;
        dualReconstruction(img, minima);
        compare(minima, img, minima, CMP_GT);
    }	

	void homotopyModification(Mat &grad, Mat &markers, Mat &hmod)
	{
		hmod = Mat(markers.size(), CV_8U, Scalar::all(255));
		for (int i = 0; i < hmod.rows; i++)
		{
			uchar *reconst_px = hmod.ptr<uchar>(i);
			int *markers_px = markers.ptr<int>(i);
			for (int j = 0; j < hmod.cols; j++, reconst_px++, markers_px++)
				if (*markers_px > 0)
					*reconst_px = 0;
		}
	
		imgseg::dualReconstruction(grad, hmod);
	}

	map<int, vector<Point>> getConnectedComponents(Mat &binImg, Mat &labels)
	{
		if (!labels.data)
			labels = Mat(binImg.size(), CV_32S, Scalar::all(0));

		vector<int> sets;
		sets.reserve(100);
		sets.push_back(0);//truco
		int rows = binImg.rows;
		int cols = binImg.cols;

		auto find_root = [&](int l)->int
		{
			int root = l;
			while(sets[root] != root)
				root = sets[root];
			//comprimiendo caminos
			int same = l;
			while(sets[same] != root)
			{
				int tmp = sets[same];
				sets[same] = root;
				same = tmp;
			}
			return root;
		};

		rectangle(binImg, Point(0,0), Point(cols-1, rows-1), Scalar(0));//truco

		//primera pasada
		int label = 1;
		for (int i = 1; i < rows; i++)
		{
			uchar *ptrBinImg = binImg.ptr<uchar>(i) + 1;
			uchar *ptrPrevBinImg = binImg.ptr<uchar>(i - 1) + 1;

			int *ptrLabels = labels.ptr<int>(i) + 1;
			int *ptrPrevLabels = labels.ptr<int>(i - 1) + 1;

			for (int j = 1; j < cols; j++, ptrBinImg++, ptrPrevBinImg++, ptrLabels++, ptrPrevLabels++)
			{
				if (*ptrBinImg > 0)
				{
					int up = (*ptrPrevBinImg > 0)? find_root(*ptrPrevLabels) : INT_MAX;
					int left = (*(ptrBinImg - 1) > 0)? find_root(*(ptrLabels - 1)) : INT_MAX;

					int root = std::min(up, left);

					if (root == INT_MAX)//no hay vecinos en el foreground
					{
						*ptrLabels = label;
						sets.push_back(label++);
					}
					else
					{
						*ptrLabels = root;
						if (up != INT_MAX && left != INT_MAX)
						{
							if (up < left)
								sets[left] = root;
							else
								sets[up] = root;
						}
					}
				}
			}
		}

		//segunda pasada
		map<int, vector<Point>> connCompts;
		int labelCount = 1;
		vector<int> finalLabels(label + 1, 0);
		for (int i = 1; i < rows; i++)
		{
			int *ptrLabels = labels.ptr<int>(i) + 1;
			for (int j = 1; j < cols; j++, ptrLabels++)
				if (*ptrLabels > 0)
				{
					int set = find_root(*ptrLabels);
					if(!finalLabels[set])
						finalLabels[set] = labelCount++;
					
					int finalLbl = finalLabels[set];
					*ptrLabels = finalLbl;
					connCompts[finalLbl].push_back(Point(i, j));
				}
		}
		return connCompts;
	}	

	void removeConnCompts(map<int, vector<Point>> &connCompts, int minSize)
	{
		vector<Point>::size_type msize = static_cast<vector<Point>::size_type>(minSize);
		vector<int> remove(connCompts.size());
		auto itmap = connCompts.begin();
		auto itendmap = connCompts.end();
		for(; itmap != itendmap; itmap++)
			if((itmap->second).size() < msize)
				remove.push_back(itmap->first);
		//eliminando
		auto itvec = remove.begin();
		auto itvecend = remove.end();
		for(; itvec != itvecend; itvec++)
			connCompts.erase(*itvec);
	}

	void getMarkers(Mat &colorImg, Mat &binImg, Mat &markers, Mat &se, Point &anchor, int minConnComptSize, int colorDist, int hsep, int vsep)
	{
		//hsep y vsep tienen que ser > 0;

		const int rows = colorImg.rows;
		const int cols = colorImg.cols;
		const int hDist = hsep + se.cols;
		const int vDist = vsep + se.rows;

		int label = 1;//las etiquetas que se van generando
		map<int, Vec3b> markerColor;//almacena por cada nuevo label el color que lo determino
		map<int, vector<int>> connComptMarkers;//almacena por cada componente conexa una lista de los labels que posee

		Mat labels(colorImg.size(), CV_32S);
		map<int, vector<Point>> connCompts = getConnectedComponents(binImg, labels);
		removeConnCompts(connCompts, minConnComptSize);

		vector<Point> neighb;//vector de direcciones
		for (int i = 0; i < se.rows; i++)
		{
			uchar *p = se.ptr<uchar>(i);
			for (int j = 0; j < se.cols; j++, p++)
				if(*p > 0)
					neighb.push_back(Point(i - anchor.x, j - anchor.y));
		}

		Mat seValidator(se.rows + 2*vsep, se.cols + 2*hsep, CV_8U);
		rectangle(seValidator, Point(hsep, 0), Point(hsep + se.cols, seValidator.rows - 1), Scalar(255), CV_FILLED);
		rectangle(seValidator, Point(0, vsep), Point(seValidator.cols - 1, vsep + se.rows), Scalar(255), CV_FILLED);
		vector<Point> seNeighbValidator;
		Point seValidatorAnchor(seValidator.rows / 2, seValidator.cols / 2);
		for (int i = 0; i < seValidator.rows; i++)
		{
			uchar *valid = seValidator.ptr<uchar>(i);
			for (int j = 0; j < seValidator.cols; j++, valid++)
				if(*valid)
					seNeighbValidator.push_back(Point(i - seValidatorAnchor.x, j - seValidatorAnchor.y));
		}

		auto validPos = [&](int x, int y)
		{
			return (x >= 0 && x < rows && y >= 0 && y < cols);
		};

		auto cc_it = connCompts.begin();
		auto cc_itend = connCompts.end();
		for (; cc_it != cc_itend; cc_it++, label++)//por cada componente conexa
		{
			vector<Point> points = cc_it->second;
			auto pt_it = points.begin();
			auto pt_itend = points.end();					

			for (; pt_it != pt_itend; pt_it++)//por cada pixel p en la componente conexa
			{
				//verificar que el punto mantiene buena distancia con los demas marcadores
				bool nearSe = false;
				for each (Point p in seNeighbValidator)
				{
					int dx = pt_it->x + p.x;
					int dy = pt_it->y + p.y;
					if(validPos(dx, dy) && *(markers.ptr<int>(dx) + dy) > 0 && *(labels.ptr<int>(dx) + dy) == cc_it->first)
					{
						nearSe = true;
						break;
					}
				}
				if (nearSe)
					continue;

				auto dir_it = neighb.begin();
				auto dir_itend = neighb.end();

				bool contained = true;
				bool oneColor = true;

				Vec3b color;
				bool colorSelected = false;

				for (; dir_it != dir_itend; dir_it++)//por cada vecino de p de acuerdo a se
				{
					int dx = pt_it->x + dir_it->x;
					int dy = pt_it->y + dir_it->y;

					if (validPos(dx, dy))
					{
						if(!*(binImg.ptr<uchar>(dx) + dy))//el se no esta completamente contenido en la componente conexa
						{
							contained = false;
							break;
						}

						Vec3b neighbColor = *(colorImg.ptr<Vec3b>(dx) + dy);
						if (colorSelected)//si ya se consiguio un color para analizar
						{							
							int cdist = colorDistance(color, neighbColor);

							if (cdist > colorDist)//dentro del se no hay colores 'iguales'
							{
								oneColor = false;
								break;
							}
						}
						else
						{
							colorSelected = true;
							color = neighbColor;
						}
					}
					else//el se no esta contenido en la imagen
					{
						contained = false;
						break;
					}
				}

				if(contained && oneColor)//si se estaba completamente contenido y habia un solo color, crear marcador (puede que ya tenga una etiqueta)
				{
					//buscar si este color fue etiquetado en esta componente conexa
					auto createdLabel_it = (connComptMarkers[cc_it->first]).begin();
					auto createdLabel_itend = (connComptMarkers[cc_it->first]).end();

					int lbl = 0;
					for (; createdLabel_it != createdLabel_itend; createdLabel_it++)
					{
						Vec3b labeledColor = markerColor.at(*createdLabel_it);
						int cdist = colorDistance(color, labeledColor);
						
						if(cdist <= colorDist)//ya existia un label para ese color
						{
							lbl = *createdLabel_it;
							break;
						}
					}

					if(!lbl)//si no existia un label para ese color crear uno nuevo y llenar datos en markerColor y connComptMarkers
					{
						lbl = label++;
						(connComptMarkers.at(cc_it->first)).push_back(lbl);
						markerColor[lbl] = color;
					}					

					//ahora crear el marcador en markers
					dir_it = neighb.begin();
					dir_itend = neighb.end();

					for (; dir_it != dir_itend; dir_it++)//por cada vecino de p de acuerdo a se
					{
						int dx = pt_it->x + dir_it->x;
						int dy = pt_it->y + dir_it->y;

						//no voy a verificar posicion valida porque se asume que el se esta contenido
						*(markers.ptr<int>(dx) + dy) = lbl;
					}
				}
			}
		}
	}

	int inline colorDistance(Vec3b &c1, Vec3b &c2)
	{
		//int l = std::abs(static_cast<int>(c1[0]) - static_cast<int>(c2[0]));
		//int a = std::abs(static_cast<int>(c1[1]) - static_cast<int>(c2[1]));
		//int b = std::abs(static_cast<int>(c1[2]) - static_cast<int>(c2[2]));
		//return l + a + b;//manhattan

		return static_cast<int>(norm<int, 3>(Vec3i(c1[0] - c2[0],
												   c1[1] - c2[1],
												   c2[2] - c2[2])));//euclideana	
	}

	void fillRegions(Mat &labels, Mat &regions, Vec3b lineColor)
	{
		if (!regions.data)
			regions = Mat(labels.size(), CV_8UC3);

		int rows = labels.rows;
		int cols = labels.cols;
		map<int, Vec3b> color;

		auto it = labels.begin<int>();
		auto itend = labels.end<int>();
		auto itcolor = regions.begin<Vec3b>();

		for (; it != itend; it++, itcolor++)
			if (*it > 0)
			{
				if (color.count(*it))
					*itcolor = color.at(*it);
				else
				{
					Vec3b c(rand()%255, rand()%255, rand()%255);
					color[*it] = c;
					*itcolor = c;
				}
			}
			else
				*itcolor = lineColor;
	}

	void toGrayScale(Mat &src, Mat &dst)
	{
		//src y dst tienen que tener el mismo size
		//el tipo de src=CV_8UC3 y el de dst=CV_8U
		if (dst.empty())
			dst = Mat(src.size(), CV_8U);
		
		auto it = src.begin<Vec3b>();
		auto itend = src.end<Vec3b>();
		auto dstit = dst.begin<uchar>();
		for (; it != itend; it++, dstit++)
		{
			Vec3b color = *it;
			//metodo de desaturacion Gray = ( Max(Red, Green, Blue) + Min(Red, Green, Blue) ) / 2
			//uchar cmax = std::max<uchar>(color[0], color[1]);
			//cmax = std::max<uchar>(color[2], cmax);
			//uchar cmin = std::min<uchar>(color[0], color[1]);
			//cmin = std::min<uchar>(color[2], cmin);
			//*dstit = (cmax + cmin) / 2;

			//average
			//int avg = (color[0] + color[1] + color[2]) / 3;
			//*dstit = static_cast<uchar>(avg);

			//otra (Red * 0.3 + Green * 0.59 + Blue * 0.11)
			float avg = (0.11f*color[0] + 0.59f*color[1] + 0.3f*color[2]);
			*dstit = static_cast<uchar>(avg);
		}
	}

	void bitMixingImg(Mat &src, Mat &dst)
	{
		auto itSrc = src.begin<Vec3b>();
		auto itendSrc = src.end<Vec3b>();
		auto itDst = dst.begin<int>();

		for (; itSrc != itendSrc; itSrc++, itDst++)
		{
			int b = (*itSrc)[0];
			int g = ((*itSrc)[0]) << 8;
			int r = ((*itSrc)[0]) << 16;

			*itDst = (b | g | r);
		}
	}
}
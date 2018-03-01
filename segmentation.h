#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace imgseg
{
	//Watershed basado en distancia geodesica
	//grad: gradiente de la imagen.
	//markers: marcadores (enteros mayores que 0) iniciales por cada 
	//objeto de la imagen que se desea segmentar. 
	//Al finalizar el algoritmo markers tendra valor -1 en los pixeles 
	//que pertenecen a las lineas de watershed y en el resto tendra las
	//etiquetas correspondientes de cada region segmentada (objetos).
	void gradWshed(cv::Mat &grad, cv::Mat &markers);

	//Watershed basado en distancia geodesica
	//img: imagen original (el gradiente se aproxima en el algoritmo)
	//markers: marcadores (enteros mayores que 0) iniciales por cada 
	//objeto de la imagen que se desea segmentar. 
	//Al finalizar el algoritmo markers tendra valor -1 en los pixeles 
	//que pertenecen a las lineas de watershed y en el resto tendra las
	//etiquetas correspondientes de cada region segmentada (objetos).
	void watershed(cv::Mat &img, cv::Mat &markers);

	//Watershed basado en crecimiento de regiones
	//img: imagen original en colores. No se calcula el gradiente, se 
	//emplea una similitud entre los colores de los píxeles.
	//markers: marcadores (enteros mayores que 0) iniciales por cada 
	//objeto de la imagen que se desea segmentar. 
	//Al finalizar el algoritmo, en los pixeles de markers quedaran valores positivos
	//correspondientes a las etiquetas en cada region segmentada (objetos).
	void watershedColor(cv::Mat &img, cv::Mat &markers);

	//Watershed con viscosidad. Es mas tolerante al ruido y puede
	//completar contornos incompletos. Esta es otra version de lo 
	//implementado en viscousForceWatershed(cv::Mat &img, cv::Mat &markers).
	//img: imagen original.
	//markers: marcadores o etiquetas (enteros mayores que 0) iniciales por cada 
	//objeto de la imagen que se desea segmentar. 
	//Al finalizar el algoritmo, en los pixeles de markers quedaran valores positivos
	//correspondientes a las etiquetas en cada region segmentada (objetos).
	void vfWatershed(cv::Mat &img, cv::Mat &markers);

	//Watershed basado en distancia topografica, que utiliza la modificacion del 
	//algoritmo de Moore en el proceso de inundacion
	//grad: gradiente de la imagen (con modificacion homotopica segun los marcadores).
	//markers: marcadores (enteros mayores que 0) iniciales por cada 
	//objeto de la imagen que se desea segmentar.
	void topographicDistWshed(cv::Mat &grad, cv::Mat &markers);

	//Reconstruccion geodesica dual (por erosion) de la  
	//imagen mask a partir de la imagen marcador marker.
	void dualReconstruction(cv::Mat &mask, cv::Mat &marker);
		
	//Determina los minimos de la imagen img con dynamics mayor que h, 
	//y se almacenan en la imagen binaria minima (background = 0, foreground > 0).
	void getMinima(cv::Mat &img, cv::Mat &minima, int h=1);

	//Modificacion homotopica del gradiente a partir de marcadores
	void homotopyModification(cv::Mat &grad, cv::Mat &markers, cv::Mat &hmod);

	//Determina las componentes conexas de la imagen binaria binImg (background = 0, foreground > 0)
	//etiquetandolas con enteros mayores que 0 en labels.
	std::map<int, std::vector<cv::Point>> getConnectedComponents(cv::Mat &binImg, cv::Mat &labels);

	//Del diccionario de componentes conexas que devuelve connectedComponents(cv::Mat &binImg, cv::Mat &labels)
	//elimina las menores que minSize.
	void removeConnCompts(std::map<int, std::vector<cv::Point>> &connCompts, int minSize=40);

	//Devuelve marcadores en markers a partir de las componentes conexas de 
	//binImg teniendo en cuenta los colores en colorImg.
	//se:: elemento estructurante que determina la forma de los marcadores.
	//anchor: centro de se.
	//minConnComptSize: cantidad minima de pixeles para considerar una componente conexa en el procesamiento.
	//colorDist: distancia maxima permitida para considerar dos colores iguales.
	//hsep: separacion horizontal entre elementos estructurantes.
	//vsep: separacion vertical entre elementos estructurantes.
	void getMarkers(cv::Mat &colorImg, cv::Mat &binImg, cv::Mat &markers, cv::Mat &se, cv::Point &anchor, int minConnComptSize, int colorDist=10, int hsep=10, int vsep=10);

	//Establece una similitud entre dos colores.
	inline int colorDistance(cv::Vec3b &c1, cv::Vec3b &c2);

	//Pinta cada region etiquetada en labels con un color aleatorio en regions
	//las etiquetas con valores no positivos tomaran el valor de lineColor
	void fillRegions(cv::Mat &labels, cv::Mat &regions, cv::Vec3b lineColor=cv::Vec3b(0,0,0));

	//Alternativas para convertir a escala de grises.
	void toGrayScale(cv::Mat &src, cv::Mat &dst);

	//Codificar colores en un solo canal de enteros entre 0 y 2^24 - 1
	void bitMixingImg(cv::Mat &src, cv::Mat &dst);
}

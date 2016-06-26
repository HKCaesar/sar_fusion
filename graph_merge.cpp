//
// Created by auroua on 16-6-24.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <string.h>
#include <map>
#include <list>
#include <set>
#include <math.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iterator>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

inline double square( double a )
{
	return a*a;
}


inline double diff( const cv::Mat &img, int x1, int y1, int x2, int y2 )
{
	return sqrt( square( img.at<cv::Vec3f>( y1, x1 )[0] - img.at<cv::Vec3f>( y2, x2 )[0] ) +
				 square( img.at<cv::Vec3f>( y1, x1 )[1] - img.at<cv::Vec3f>( y2, x2 )[1] ) +
				 square( img.at<cv::Vec3f>( y1, x1 )[2] - img.at<cv::Vec3f>( y2, x2 )[2] ) );
}


struct UniverseElement
{
	int rank;
	int p;
	int size;

	UniverseElement() : rank( 0 ), size( 1 ), p( 0 ) {}
	UniverseElement( int rank, int size, int p ) : rank( rank ), size( size ), p( p ) {}
};

struct edge
{
	int a;
	int b;
	double w;
};


bool operator<( const edge &a, const edge &b )
{
	return a.w < b.w;
}

class Universe
{
private:
	std::vector<UniverseElement> elements;
	int num;

public:

	Universe( int num ) : num( num )
	{
		elements.reserve( num );

		for ( int i = 0; i < num; i++ )
		{
			elements.emplace_back( 0, 1, i );
		}
	}

	~Universe() {}

	int find( int x )
	{
		int y = x;
		while ( y != elements[y].p )
		{
			y = elements[y].p;
		}
		elements[x].p = y;

		return y;
	}

	void join( int x, int y )
	{
		if ( elements[x].rank > elements[y].rank )
		{
			elements[y].p = x;
			elements[x].size += elements[y].size;
		}
		else
		{
			elements[x].p = y;
			elements[y].size += elements[x].size;
			if ( elements[x].rank == elements[y].rank )
			{
				elements[y].rank++;
			}
		}
		num--;
	}

	int size( int x ) const { return elements[x].size; }
	int numSets() const { return num; }
};

double calThreshold( int size, double scale )
{
	return scale / size;
}

void visualize( const cv::Mat &img, std::shared_ptr<Universe> universe )
{
	const int height = img.rows;
	const int width = img.cols;
	std::vector<cv::Vec3b> colors;

	cv::Mat segmentated( height, width, CV_8UC3 );

	std::random_device rnd;
	std::mt19937 mt( rnd() );
	std::uniform_int_distribution<> rand256( 0, 255 );

	for ( int i = 0; i < height*width; i++ )
	{
		cv::Vec3b color( rand256( mt ), rand256( mt ), rand256( mt ) );
		colors.push_back( color );
	}

	for ( int y = 0; y < height; y++ )
	{
		for ( int x = 0; x < width; x++ )
		{
			segmentated.at<cv::Vec3b>( y, x ) = colors[universe->find( y*width + x )];
		}
	}

	cv::imshow( "Initial Segmentation Result", segmentated );
	cv::waitKey( 0 );
}

std::shared_ptr<Universe> segmentGraph( int numVertices, int numEdges, std::vector<edge> &edges, double scale )
{
	std::sort( edges.begin(), edges.end() );

	auto universe = std::make_shared<Universe>( numVertices );

	std::vector<double> threshold( numVertices, scale );

	for ( auto &pedge : edges )
	{
		int a = universe->find( pedge.a );
		int b = universe->find( pedge.b );

		if ( a != b )
		{
			if ( ( pedge.w <= threshold[a] ) && ( pedge.w <= threshold[b] ) )
			{
				universe->join( a, b );
				a = universe->find( a );
				threshold[a] = pedge.w + calThreshold( universe->size( a ), scale );
			}
		}
	}

	return universe;
}


// image segmentation using "Efficient Graph-Based Image Segmentation"
std::shared_ptr<Universe> segmentation( const cv::Mat &img, double scale, double sigma, int minSize )
{
	const int width = img.cols;
	const int height = img.rows;

	cv::Mat imgF;
	img.convertTo( imgF, CV_32FC3 );

	cv::Mat blurred;
	cv::GaussianBlur( imgF, blurred, cv::Size( 5, 5 ), sigma );

	std::vector<edge> edges( width*height * 4 );

	int num = 0;
	for ( int y = 0; y < height; y++ )
	{
		for ( int x = 0; x < width; x++ )
		{
			if ( x < width - 1 )
			{
				edges[num].a = y * width + x;
				edges[num].b = y * width + ( x + 1 );
				edges[num].w = diff( blurred, x, y, x + 1, y );
				num++;
			}

			if ( y < height - 1 )
			{
				edges[num].a = y * width + x;
				edges[num].b = ( y + 1 ) * width + x;
				edges[num].w = diff( blurred, x, y, x, y + 1 );
				num++;
			}

			if ( ( x < width - 1 ) && ( y < height - 1 ) )
			{
				edges[num].a = y * width + x;
				edges[num].b = ( y + 1 ) * width + ( x + 1 );
				edges[num].w = diff( blurred, x, y, x + 1, y + 1 );
				num++;
			}

			if ( ( x < width - 1 ) && ( y > 0 ) )
			{
				edges[num].a = y * width + x;
				edges[num].b = ( y - 1 ) * width + ( x + 1 );
				edges[num].w = diff( blurred, x, y, x + 1, y - 1 );
				num++;
			}
		}
	}

	auto universe = segmentGraph( width*height, num, edges, scale );


	for ( int i = 0; i < num; i++ )
	{
		int a = universe->find( edges[i].a );
		int b = universe->find( edges[i].b );
		if ( ( a != b ) && ( ( universe->size( a ) < minSize ) || ( universe->size( b ) < minSize ) ) )
		{
			universe->join( a, b );
		}
	}

	return universe;
}

std::shared_ptr<Universe> generateSegments( const cv::Mat &img, double scale = 3, double sigma = 0.8, int minSize = 10 )
{
	auto universe = segmentation( img, scale, sigma, minSize );

	visualize( img, universe );

	return universe;
}



int main(){
	string input_url = "/home/auroua/workspace/matlab2015/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TRAIN/17_DEG/T72/SN_132/HB03882.015.jpg";
	string output_url = "/home/auroua/workspace/output2.png";
	cv::Mat img_input = cv::imread(input_url, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	auto universe = generateSegments( img_input);
}
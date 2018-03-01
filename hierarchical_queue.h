#pragma once
#include <deque>
#include <vector>
#include <array>
#include <memory>
#include <map>
#include <queue>

using namespace std;

namespace imgseg
{
	/*************************************************************/	
	/****   Colas jerarquicas. Implementaciones eficientes.   ****/
	/*************************************************************/	
	/****HQueue esta implementada para una constante prefijada****/
	/****de jerarquias en los enteros, mientras que MapHQueue ****/
	/****es mas generica en cuanto a la cantidad y el tipo de ****/
	/****dato de la jerarquia.                                ****/
	/*************************************************************/

    template <class T, int H>
    class HQueue
    {
    private:  

		struct Node
		{
			T value;
			Node *next;

			Node(){ next = nullptr; }
		};

        array<Node*, H> _first;
        array<Node*, H> _last;        
        vector<Node> _value;
        
        int _current_h;
        int _count;
        int _index;
        int _capacity;

        bool _locked;        

    public:
        HQueue(int capacity)
        {
			_value = vector<Node>(capacity);

            for(int i = 0; i < H; i++)
				_first[i] = nullptr;

            _current_h = H;//0 <= _current_h < H
            _count = 0;
            _index = -1;
            _capacity = capacity;
            _locked = false;
        }

        void Push(T value, int h)
        {
            if(h < 0 || h >= H)
                throw "hierarchy parameter out of range";

            if(_index >= _capacity)
                throw "index out of range";

			_count++;
			_index++;

			Node *n = &(_value[_index]);
			n->value = value;

            if(h < _current_h)        
                if(_locked)
                {                    
                    Node *last_node = _last[_current_h];
					last_node->next = n;
					_last[_current_h] = n;	
                    return;
                }
                else
                    _current_h = h;
        
            if (!_first[h])
				_first[h] = _last[h] = n;
			else
			{
				Node *last_node = _last[h];
				last_node->next = n;
				_last[h] = n;					
			}
        }

        T Pop()
        {        
            if(_count == 0)
                throw "queue empty before pop";

			Node *firstNode = _first[_current_h];

			if(!firstNode->next)
				while (!_first[++_current_h]);
			else
				_first[_current_h] = firstNode->next;

            _locked = true;			
            _count--;

			return firstNode->value;
        }

        bool Empty()    
        {
            return _count == 0;
        }
    };  

	template <class T, class H>
    class MapHQueue
    {
    private:      

		struct Node
		{
			T value;
			Node *next;

			Node(){ next = nullptr; }
		};

		map<H, pair<Node*,Node*>> _hqueues;
        vector<Node> _value;
        
		H _current_h;
        bool _locked;
		int _count;
        int _index;
        int _capacity;

    public:
        MapHQueue(int capacity)
        {
			_value = vector<Node>(capacity);    
            _count = 0;
			_index = -1;
            _capacity = capacity;
            _locked = false;
        }

        void Push(T value, H h)
        {
            if(_index >= _capacity)
				throw "the queue capacity is full";

			if(_count++ == 0)
				_current_h = h;

			Node *n = &(_value[++_index]);
			n->value = value; 

            if(h < _current_h)        
                if(_locked)
                {                    
					pair<Node*, Node*> *p = &(_hqueues.begin()->second);
					(p->second)->next = n;
					p->second = n;
                    return;
                }
                else
                    _current_h = h; 

			if (!_hqueues.count(h))
				_hqueues[h] = pair<Node*,Node*>(n, n);
			else
			{
				pair<Node*, Node*> *p = &(_hqueues.at(h));
				(p->second)->next = n;
				p->second = n;
			}
        }

        T Pop()
        {        
            if(_count == 0)
                throw "queue empty before pop";

			pair<Node*, Node*> *p = &(_hqueues.begin()->second);
			Node *firstNode = p->first;

			if(!firstNode->next)
			{
				_hqueues.erase(_current_h);
				_current_h = (_hqueues.begin())->first;
			}
			else
				p->first = firstNode->next;

            _locked = true;			
            _count--;

			return firstNode->value;
        }

        bool Empty()    
        {
            return _count == 0;
        }
    };

	/*******************************************************/
	/*******************************************************/
	/*******************************************************/
	
	
	/*Las siguientes son implementaciones de PRUEBA de colas jerarquicas*/

	template <class T, class H>
    class OrderedQueue
    {
    private:
		map<H, deque<T>> _hqueue;
		H _current_h;
		H _max_h;
        int _count;
        bool _locked;        

    public:
        OrderedQueue()
        {
            _count = 0;
            _locked = false;
        }

        void Push(T value, H h)
        {
			if(_count == 0)
				_current_h = _max_h = h;

			if (h < _current_h)
			{
				if (_locked)
				{
					(_hqueue.at(_current_h)).push_back(value);

					_count++;
					return;
				}
				else
					_current_h = h;
			}
			else if(h > _max_h)
				_max_h = h;

			deque<T> *queue_h = &(_hqueue[h]);
			
			queue_h->push_back(value);
			_count++;
        }

        T Pop()
        {        
            if(_count == 0)
                throw "queue empty before pop";

			deque<T> *currentq = &(_hqueue.at(_current_h));
			
			if (currentq->empty())
			{
				_hqueue.erase(_current_h);

				_current_h = _max_h;
				auto it = _hqueue.begin();
				auto itend =  _hqueue.end();				
				for (; it != itend; it++)
					if(it->first < _current_h)
						_current_h = it->first;

				currentq = &(_hqueue.at(_current_h));
			}

			T result = currentq->front();
			currentq->pop_front();
			_count--;

			return result;
        }

        bool Empty()    
        {
            return _count == 0;
        }

		void SetLock(bool lock)
		{
			_locked = lock;
		}
    };

	template <class T>
    class hierarchical_queue
    {

    private:

        deque<T>** _queue_list;
    
        int _current_h;
        int _hmax;
        int _count;

        bool _locked;

        void initialize(int h)
        {
            _queue_list = new deque<T>*[h];

            for(int i = 0; i < h; i++)
                _queue_list[i] = new deque<T>();
            
            _current_h = INT_MAX;
            _hmax = h;
            _count = 0;
            _locked = false;
        }

        void destroy()
        {
            for(int i = 0; i < _hmax; i++)
                delete _queue_list[i];        
     
            delete [] _queue_list;
        }

    public:

        hierarchical_queue(int h)
        {
            if(h <= 0)
                throw "h must be positive";

            initialize(h);
        }

        hierarchical_queue()
        {
            initialize(256);
        }

        ~hierarchical_queue()
        {
            destroy();
        }

        void push(T value, int h)
        {
            if(h < 0 || h >= _hmax)
                throw "hierarchy parameter out of range";

            _count++;

            if(h < _current_h)        
                if(_locked)
                {
                    _queue_list[_current_h]->push_back(value);
                    return;
                }
                else
                    _current_h = h;
        
            _queue_list[h]->push_back(value);
        }

        T pop()
        {        
            if(_count == 0)
                throw "queue empty before pop";

            while(_queue_list[_current_h]->empty())
                _current_h++;

            _locked = true;
        
            T result = _queue_list[_current_h]->front();
            _queue_list[_current_h]->pop_front();
            _count--;

            return result;
        }

        T top()
        {
            if(_count == 0)
                throw "queue empty before top";

            int top_h = _current_h;

            while(_queue_list[top_h]->empty())
                top_h++;
        
            return _queue_list[top_h]->front();
        }

        bool empty()    
        {
            return _count == 0;
        }

        int count()
        {
            return _count;
        }    

        void reset()
        {
            destroy();
            initialize(_hmax);
        }
    };
}